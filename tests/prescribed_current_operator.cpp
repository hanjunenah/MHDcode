// Test the production current constraint and SSPRK stages on small meshes.
// Standard headers precede the access macro used for the solver translation unit.
#include <hdf5.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define private public
#define main mhd_solver_program_main
#include "../src/main.cpp"
#undef main
#undef private

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

double maximum_difference(const std::vector<double>& a, const std::vector<double>& b) {
    check(a.size() == b.size(), "Mismatched test vectors");
    double result = 0.0;
    for (std::size_t n = 0; n < a.size(); ++n)
        result = std::max(result, std::abs(a[n] - b[n]));
    return result;
}

void check_manufactured_constraint(bool diagonal) {
    const int nx = 27, ny = 21;
    const double dx = 0.05, dy = 0.04;
    std::vector<double> solid(nx * ny, 0.0);
    for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
        const bool inside = diagonal ?
            (i >= 5 && i <= 21 && j == 4 + (i - 5) / 2) :
            (i >= 7 && i <= 17 && j >= 5 && j <= 14);
        if (inside) solid[j * nx + i] = 1.0;
    }
    PrescribedCurrentOperator op(nx, ny, dx, dy, solid);
    std::vector<double> reference(op.node_count());
    for (int j = 0; j <= ny; ++j) for (int i = 0; i <= nx; ++i) {
        const double x = i * dx, y = j * dy;
        reference[op.node_index(i, j)] =
            0.015 * std::sin(7.0 * x) * std::cos(5.0 * y) +
            0.004 * x * y + 0.003 * x - 0.002 * y;
    }
    auto target = op.current(reference);
    auto corrected = reference;
    for (std::size_t n = 0; n < corrected.size(); ++n) {
        if (op.powered_mask()[n]) corrected[n] += 0.01 * std::cos(0.37 * n);
        else target[n] = 0.0;
    }
    const auto old = corrected;
    op.enforce(corrected, target);
    check(maximum_difference(corrected, reference) < 2e-11,
          "Manufactured Poisson solution was not recovered inside the powered body");
    check(op.max_residual(corrected, target) < 2e-9,
          "Actual dual current differs from its prescribed value");
    for (std::size_t n = 0; n < corrected.size(); ++n)
        if (!op.powered_mask()[n]) check(corrected[n] == old[n],
            "Current constraint overwrote a plasma potential node");
    const auto once = corrected;
    op.enforce(corrected, target);
    check(maximum_difference(corrected, once) < 2e-13,
          "Repeated current enforcement is not idempotent");

    std::vector<double> bxf, byf;
    op.face_fields(corrected, bxf, byf);
    double divergence = 0.0;
    for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
        const double div = (bxf[j * (nx + 1) + i + 1] - bxf[j * (nx + 1) + i]) / dx +
            (byf[(j + 1) * nx + i] - byf[j * nx + i]) / dy;
        divergence = std::max(divergence, std::abs(div));
    }
    check(divergence < 2e-13, "Native field curl introduced magnetic divergence");

    // A local change of prescribed current must obey Faraday through a local
    // Ez=-dAz/dt, leaving every nonpowered plasma potential unchanged.
    std::vector<double> a0(op.node_count(), 0.0), a1 = a0;
    std::vector<double> cell_command(nx * ny, 0.0);
    for (std::size_t c = 0; c < solid.size(); ++c)
        if (solid[c]) cell_command[c] = 0.2 * (1.0 + 0.1 * std::sin(0.2 * c));
    const auto j1 = op.project_current(cell_command);
    double cell_integral = 0.0, node_integral = 0.0;
    for (const double value : cell_command) cell_integral += value * dx * dy;
    for (int j = 0; j <= ny; ++j) for (int i = 0; i <= nx; ++i) {
        double expected = 0.0;
        for (int jj = j - 1; jj <= j; ++jj) for (int ii = i - 1; ii <= i; ++ii)
            if (ii >= 0 && ii < nx && jj >= 0 && jj < ny)
                expected += 0.25 * cell_command[jj * nx + ii];
        const auto node = op.node_index(i, j);
        check(std::abs(j1[node] - expected) < 2e-16,
              "Cell-to-dual current projection is not the declared quarter sum");
        node_integral += j1[node] * dx * dy;
    }
    check(std::abs(cell_integral - node_integral) < 2e-14,
          "Current projection changed the integrated prescribed current");
    op.enforce(a1, j1);
    for (std::size_t n = 0; n < a1.size(); ++n)
        if (!op.powered_mask()[n]) check(a1[n] == 0.0,
            "Zero-eta feed bypassed plasma shielding with a global field solve");
    check(op.max_residual(a1, j1) < 2e-9,
          "A zero-field coil did not acquire its actual supplied current");
    const double duration = 0.17;
    auto electric = a1;
    for (std::size_t n = 0; n < electric.size(); ++n)
        electric[n] = -(a1[n] - a0[n]) / duration;
    std::vector<double> e_curl_x, e_curl_y;
    op.face_fields(electric, e_curl_x, e_curl_y);
    op.face_fields(a1, bxf, byf);
    double field_amplitude = 0.0;
    for (std::size_t f = 0; f < bxf.size(); ++f) {
        check(std::abs(bxf[f] + duration * e_curl_x[f]) < 2e-14,
              "Supplied current changed Bx inconsistently with Faraday");
        field_amplitude = std::max(field_amplitude, std::abs(bxf[f]));
    }
    for (std::size_t f = 0; f < byf.size(); ++f) {
        check(std::abs(byf[f] + duration * e_curl_y[f]) < 2e-14,
              "Supplied current changed By inconsistently with Faraday");
        field_amplitude = std::max(field_amplitude, std::abs(byf[f]));
    }
    check(field_amplitude > 1e-4, "Supplied current failed to establish magnetic field");

    // This is the native, independently calculated electrical-work identity.
    // With all plasma/exterior potentials unchanged, only the powered source
    // does work. It must equal the change of native face magnetic energy.
    double work = 0.0;
    for (std::size_t n = 0; n < a1.size(); ++n)
        if (op.powered_mask()[n]) work += 0.5 * j1[n] * (a1[n] - a0[n]) * dx * dy;
    const double energy = op.magnetic_energy(a1) - op.magnetic_energy(a0);
    check(energy > 0.0, "Powered current produced no magnetic energy");
    check(std::abs(energy - work) < 2e-13,
          "Native magnetic energy does not match supplied-current electrical work");
    std::cout << (diagonal ? "diagonal" : "rectangular")
              << " body: current residual=" << op.max_residual(a1, j1)
              << ", divB=" << divergence << ", work=" << work << '\n';
}

void check_actual_runge_kutta_stages() {
    Config c;
    c.nx = 24; c.ny = 18; c.lx = 1.0; c.ly = 0.75;
    c.boundary_x = c.boundary_y = "outflow";
    c.antenna_model = "prescribed_current";
    c.requested_threads = 1; c.t_end = 0.02;
    c.resistivity = 0.0;
    const std::size_t count = c.nx * c.ny;
    InitialData initial;
    initial.rho.assign(count, 1.0); initial.p.assign(count, 0.3);
    initial.u.assign(count, 0.0); initial.v = initial.u; initial.w = initial.u;
    initial.bx = initial.u; initial.by = initial.u; initial.bz = initial.u;
    std::vector<double> solid(count, 0.0);
    AntennaData antenna;
    antenna.time = {0.0, c.t_end};
    antenna.jx.assign(2 * count, 0.0); antenna.jy = antenna.jx; antenna.jz = antenna.jx;
    for (int j = 6; j <= 11; ++j) for (int i = 8; i <= 15; ++i) {
        const auto cell = static_cast<std::size_t>(j * c.nx + i);
        solid[cell] = 1.0;
        antenna.jz[count + cell] = 0.15 * std::sin(3.141592653589793 * (i - 7.5) / 8.0) *
            std::sin(3.141592653589793 * (j - 5.5) / 6.0);
    }
    Solver solver(c, initial, antenna, solid);
    auto& op = *solver.current_operator_;
    const double dt = 0.003;
    const auto j0 = solver.commanded_current(0.0), j1 = solver.commanded_current(dt);
    std::vector<double> jdot(j0.size()), adot0, adot1;
    for (std::size_t n = 0; n < jdot.size(); ++n) jdot[n] = (j1[n] - j0[n]) / dt;
    Solver::RhsRates rates0, rates1;
    const double ch = solver.cleaning_speed(solver.cells_);
    const auto rhs0 = solver.prescribed_rhs(solver.cells_, solver.potential_, jdot, ch, 0.0, rates0, adot0);
    check(op.max_residual(adot0, jdot) < 2e-9,
          "First stage inductive voltage does not differentiate the current constraint");
    std::vector<double> rhs_bx, rhs_by;
    op.cell_fields(adot0, rhs_bx, rhs_by);
    for (std::size_t cell = 0; cell < count; ++cell)
        check(std::abs(rhs0[cell][bx] - rhs_bx[cell]) < 2e-13 &&
              std::abs(rhs0[cell][by] - rhs_by[cell]) < 2e-13,
              "First stage FV induction is inconsistent with native Faraday before resync");
    auto stage_a = solver.potential_;
    for (std::size_t n = 0; n < stage_a.size(); ++n) stage_a[n] += dt * adot0[n];
    check(op.max_residual(stage_a, j1) < 2e-9,
          "First forward-Euler stage violates the supplied actual current");
    auto stage = solver.cells_;
    for (std::size_t cell = 0; cell < count; ++cell) {
        for (std::size_t k = 0; k < nvar; ++k) stage[cell][k] += dt * rhs0[cell][k];
        solver.constrain_solid(stage[cell], cell);
    }
    std::vector<double> stage_bx, stage_by;
    op.cell_fields(stage_a, stage_bx, stage_by);
    for (std::size_t cell = 0; cell < count; ++cell)
        check(std::abs(stage[cell][bx] - stage_bx[cell]) < 2e-13 &&
              std::abs(stage[cell][by] - stage_by[cell]) < 2e-13,
              "Forward Euler stage disagrees with curl(Az) before final synchronization");
    const auto rhs1 = solver.prescribed_rhs(stage, stage_a, jdot, ch, dt, rates1, adot1);
    op.cell_fields(adot1, rhs_bx, rhs_by);
    for (std::size_t cell = 0; cell < count; ++cell)
        check(std::abs(rhs1[cell][bx] - rhs_bx[cell]) < 2e-13 &&
              std::abs(rhs1[cell][by] - rhs_by[cell]) < 2e-13,
              "Second stage FV induction is inconsistent with native Faraday before resync");
    check(op.max_residual(adot1, jdot) < 2e-9,
          "Second stage voltage does not enforce the differentiated current");
    solver.advance_prescribed_current(dt, ch, 0.0);
    check(op.max_residual(solver.potential_, j1) < 2e-9,
          "Final SSPRK state violates the supplied current");
    check(solver.max_stage_current_constraint_error_ < 2e-9,
          "Production RHS evaluated a stage with an incorrect actual current");
    std::vector<double> cell_bx, cell_by;
    op.cell_fields(solver.potential_, cell_bx, cell_by);
    double amplitude = 0.0;
    for (std::size_t cell = 0; cell < count; ++cell) {
        const auto& q = solver.cells_[cell];
        check(std::abs(q[bx] - cell_bx[cell]) < 2e-13 &&
              std::abs(q[by] - cell_by[cell]) < 2e-13,
              "SSPRK finite-volume fields drifted from native Faraday fields");
        if (solid[cell]) {
            check(q[rho] == 1.0 && q[mx] == 0.0 && q[my] == 0.0 && q[mz] == 0.0,
                  "Current driver moved or changed density of the stationary plate");
            check(std::abs(solver.primitive(q).pressure - 0.3) < 2e-14,
                  "Current driver changed the prescribed plate thermal state");
            amplitude = std::max(amplitude, std::abs(q[bx]) + std::abs(q[by]));
        }
    }
    check(amplitude > 1e-5, "eta=0 stage update did not establish the coil's field");
    check(solver.antenna_energy_injected_ > 0.0,
          "Powered increasing current has no positive electrical source work");
    check(solver.max_wall_mass_flux_ == 0.0, "Powered body leaked plasma mass");
    std::cout << "SSPRK stages: current residual=" << op.max_residual(solver.potential_, j1)
              << ", circuit work=" << solver.antenna_energy_injected_ << '\n';
}
} // namespace

int main() {
    try {
        check_manufactured_constraint(false);
        check_manufactured_constraint(true);
        check_actual_runge_kutta_stages();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}

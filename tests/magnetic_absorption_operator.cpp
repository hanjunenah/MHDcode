// Exercise the actual production split source without the HLLD update, which
// has its own divergence truncation error. Headers precede the access macro so
// only project classes expose their members in this test translation unit.
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

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check_source(const std::string& mode_x, const std::string& mode_y) {
    Config config;
    config.nx = 48; config.ny = 32;
    config.lx = 1.0; config.ly = 0.6;
    config.boundary_x = mode_x; config.boundary_y = mode_y;
    config.absorption_width = 0.15;
    config.absorption_rate = 2.0;
    config.absorption_magnetic_diffusivity = 0.2;
    config.ambient = {1.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    config.requested_threads = 1;
    const std::size_t count = static_cast<std::size_t>(config.nx * config.ny);
    InitialData initial;
    initial.rho.assign(count, 1.0); initial.p.assign(count, 0.3);
    initial.u.assign(count, 0.0); initial.v = initial.u; initial.w = initial.u;
    initial.bx = initial.u; initial.by = initial.u; initial.bz = initial.u;
    Solver solver(config, initial, {}, {});

    // A discrete curl yields a solenoidal in-plane field even on clamped edges.
    std::vector<State> potential(count);
    for (int j = 0; j < config.ny; ++j) {
        for (int i = 0; i < config.nx; ++i) {
            const double x = (i + 0.5) * solver.dx_;
            const double y = (j + 0.5) * solver.dy_;
            potential[solver.index(i, j)][psi] =
                0.01 * std::cos(9.0 * x) * std::sin(13.0 * y);
        }
    }
    for (int j = 0; j < config.ny; ++j) {
        for (int i = 0; i < config.nx; ++i) {
            const std::size_t cell = solver.index(i, j);
            State& q = solver.cells_[cell];
            q[bx] = 0.4 + solver.data_derivative(potential, i, j, psi, 1);
            q[by] = -0.2 - solver.data_derivative(potential, i, j, psi, 0);
            q[bz] = 0.1 * std::cos(0.7 * i + 0.9 * j);
            q[total_energy] += 0.5 * (q[bx]*q[bx] + q[by]*q[by] + q[bz]*q[bz]);
        }
    }
    const auto before = solver.cells_;
    auto divergence = [&](const std::vector<State>& data, int i, int j) {
        return solver.data_derivative(data, i, j, bx, 0) +
               solver.data_derivative(data, i, j, by, 1);
    };
    const double duration = 0.05 * std::min(solver.dx_*solver.dx_, solver.dy_*solver.dy_) /
                            config.absorption_magnetic_diffusivity;
    solver.apply_absorption(duration);
    double magnetic_before = 0.0, magnetic_after = 0.0, max_div_change = 0.0;
    double max_field_change = 0.0;
    for (int j = 0; j < config.ny; ++j) {
        for (int i = 0; i < config.nx; ++i) {
            const std::size_t cell = solver.index(i, j);
            const auto& old = before[cell]; const auto& now = solver.cells_[cell];
            max_div_change = std::max(max_div_change, std::abs(
                divergence(solver.cells_, i, j) - divergence(before, i, j)));
            require(std::abs(solver.primitive(now).pressure - 0.3) < 2e-14,
                    "Artificial magnetic sink must preserve thermal pressure");
            for (int component = 0; component < 3; ++component) {
                magnetic_before += 0.5 * old[bx+component]*old[bx+component];
                magnetic_after += 0.5 * now[bx+component]*now[bx+component];
                max_field_change = std::max(max_field_change,
                    std::abs(now[bx+component] - old[bx+component]));
            }
            // Curl of the sponge E can reach one derivative stencil cell
            // beyond the nonzero profile; this box excludes a two-cell halo.
            const double x = (i+0.5)*solver.dx_, y = (j+0.5)*solver.dy_;
            const bool core_x = mode_x != "absorbing" ||
                (x > config.absorption_width+2*solver.dx_ &&
                 x < config.lx-config.absorption_width-2*solver.dx_);
            const bool core_y = mode_y != "absorbing" ||
                (y > config.absorption_width+2*solver.dy_ &&
                 y < config.ly-config.absorption_width-2*solver.dy_);
            if (core_x && core_y) {
                for (int component = 0; component < 3; ++component)
                    require(now[bx+component] == old[bx+component],
                            "Magnetic sponge changed the interior beyond its stencil halo");
            }
        }
    }
    require(max_div_change < 2e-12, "Magnetic sponge introduced discrete divergence");
    require(max_field_change > 1e-5, "Nonuniform layer field was not attenuated");
    require(magnetic_after < magnetic_before, "Magnetic sponge increased integrated magnetic energy");
    const double energy_change = (magnetic_after-magnetic_before)*solver.dx_*solver.dy_;
    require(std::abs(solver.absorption_energy_exchange_-energy_change) < 2e-13,
            "Magnetic sink reservoir does not equal removed magnetic energy");
    require(std::abs(solver.absorption_mass_exchange_) < 2e-14,
            "Pure magnetic sink changed mass");

    Solver uniform(config, initial, {}, {});
    for (auto& q : uniform.cells_) {
        q[bx] = 0.4; q[by] = -0.2; q[bz] = 0.1;
        q[total_energy] += 0.5*(0.4*0.4+0.2*0.2+0.1*0.1);
    }
    uniform.apply_absorption(duration);
    for (const auto& q : uniform.cells_)
        require(q[bx] == 0.4 && q[by] == -0.2 && q[bz] == 0.1,
                "Uniform B was forced toward the ambient field");
    std::cout << mode_x << '/' << mode_y << ": max divB change=" << max_div_change
              << ", magnetic energy removed=" << -energy_change << '\n';
}

int main() {
    try {
        check_source("absorbing", "periodic");
        check_source("periodic", "absorbing");
        check_source("absorbing", "absorbing");
        check_source("absorbing", "outflow");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}

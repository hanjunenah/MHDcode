#ifndef MHD_PRESCRIBED_CURRENT_HPP
#define MHD_PRESCRIBED_CURRENT_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

// A local, powered-current constraint for a planar (Az, Jz) antenna.
//
// Az is vertex centered, with (nx + 1) * (ny + 1) entries. Its discrete curl
// gives normal face fields and an identically divergence-free magnetic field:
//   Bx(i,j) = [Az(i,j+1) - Az(i,j)] / dy,
//   By(i,j) = -[Az(i+1,j) - Az(i,j)] / dx.
// The matching current is Jz = -Laplace_h(Az) on vertex dual volumes. This
// native current, rather than a second curl of averaged cell fields, is the
// current constrained here.
//
// Every corner of a solid cell is a powered vertex. A supplied cell current
// is projected by one quarter to each corner. Thus the current integral is
// preserved exactly, while the electromagnetic body extends half a cell
// beyond the cell-centered body mask. Zero-target powered vertices are also
// constrained: a screening current cannot hide inside the powered mask.
//
// enforce() solves the Poisson equation ONLY at powered vertices, retaining
// the surrounding plasma Az as Dirichlet data. There is no equipotential or
// perfect-conductor constraint: Az is nonuniform and B can penetrate a plate.
// The same linear operation applied to (Az_dot, Jz_dot) supplies a single
// Faraday-consistent electric field Ez = -Az_dot at an integration stage.
// Resistivity deliberately does not enter this operator.
class PrescribedCurrentOperator {
public:
    PrescribedCurrentOperator(int nx, int ny, double dx, double dy,
                              const std::vector<double>& solid_mask,
                              bool periodic_x = false, bool periodic_y = false)
        : nx_(nx), ny_(ny), dx_(dx), dy_(dy),
          periodic_x_(periodic_x), periodic_y_(periodic_y),
          unique_nx_(periodic_x ? nx : nx + 1),
          unique_ny_(periodic_y ? ny : ny + 1) {
        if (nx < 2 || ny < 2 || !std::isfinite(dx) || !std::isfinite(dy)
            || dx <= 0.0 || dy <= 0.0) {
            throw std::invalid_argument("Invalid prescribed-current grid");
        }
        if (solid_mask.size() != cell_count()) {
            throw std::invalid_argument("Prescribed-current solid mask has wrong size");
        }
        cx_ = 1.0 / (dx_ * dx_);
        cy_ = 1.0 / (dy_ * dy_);
        canonical_powered_.assign(unique_node_count(), 0);
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                const double mask = solid_mask[cell_index(i, j)];
                if (!std::isfinite(mask)) {
                    throw std::invalid_argument("Nonfinite prescribed-current solid mask");
                }
                if (mask <= 0.5) continue;
                for (int dj = 0; dj <= 1; ++dj) {
                    for (int di = 0; di <= 1; ++di) {
                        const int ni = i + di;
                        const int nj = j + dj;
                        if ((!periodic_x_ && (ni == 0 || ni == nx_))
                            || (!periodic_y_ && (nj == 0 || nj == ny_))) {
                            throw std::invalid_argument(
                                "Powered antenna vertices must be off nonperiodic boundaries");
                        }
                        canonical_powered_[canonical_index(ni, nj)] = 1;
                    }
                }
            }
        }
        powered_mask_.resize(node_count());
        for (int j = 0; j <= ny_; ++j) {
            for (int i = 0; i <= nx_; ++i) {
                powered_mask_[node_index(i, j)] =
                    canonical_powered_[canonical_index(i, j)];
            }
        }
        build_factorization();
    }

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    double dx() const { return dx_; }
    double dy() const { return dy_; }
    bool periodic_x() const { return periodic_x_; }
    bool periodic_y() const { return periodic_y_; }
    std::size_t cell_count() const {
        return static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_);
    }
    std::size_t node_count() const {
        return static_cast<std::size_t>(nx_ + 1) * static_cast<std::size_t>(ny_ + 1);
    }
    std::size_t powered_node_count() const { return unknown_nodes_.size(); }
    std::size_t half_bandwidth() const { return bandwidth_; }
    std::size_t node_index(int i, int j) const {
        return static_cast<std::size_t>(j) * static_cast<std::size_t>(nx_ + 1)
             + static_cast<std::size_t>(i);
    }
    std::size_t cell_index(int i, int j) const {
        return static_cast<std::size_t>(j) * static_cast<std::size_t>(nx_)
             + static_cast<std::size_t>(i);
    }
    const std::vector<std::uint8_t>& powered_mask() const { return powered_mask_; }

    // The returned vector includes duplicate periodic seam entries. Integrals
    // should use integrated_current(), which counts a seam only once.
    std::vector<double> project_current(const std::vector<double>& cell_current) const {
        check_cells(cell_current);
        std::vector<double> result(node_count(), 0.0);
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                const double quarter = 0.25 * cell_current[cell_index(i, j)];
                if (!std::isfinite(quarter)) {
                    throw std::invalid_argument("Nonfinite prescribed cell current");
                }
                for (int dj = 0; dj <= 1; ++dj) {
                    for (int di = 0; di <= 1; ++di) {
                        result[full_index(canonical_index(i + di, j + dj))] += quarter;
                    }
                }
            }
        }
        sync_periodic(result);
        return result;
    }

    // Enforce -Laplace_h(A) = target at every powered vertex. Free plasma
    // vertices are untouched (apart from duplicate periodic seam entries).
    // This method also applies unchanged to A_dot and target_dot.
    void enforce(std::vector<double>& a, const std::vector<double>& target) const {
        check_nodes(a);
        check_nodes(target);
        const std::size_t count = unknown_nodes_.size();
        std::vector<double> rhs(count, 0.0);
        for (std::size_t k = 0; k < count; ++k) {
            const std::size_t node = unknown_nodes_[k];
            rhs[k] = target[full_index(node)];
            if (!std::isfinite(rhs[k])) {
                throw std::invalid_argument("Nonfinite prescribed nodal current");
            }
            const auto neighbors = neighbor_nodes(node);
            for (std::size_t d = 0; d < 4; ++d) {
                if (unknown_index_[neighbors[d]] < 0) {
                    rhs[k] += coefficient(d) * a[full_index(neighbors[d])];
                }
            }
            if (!std::isfinite(rhs[k])) {
                throw std::invalid_argument("Nonfinite powered-current Dirichlet data");
            }
        }
        // Fixed banded Cholesky factor: O(N * bandwidth) work for each stage.
        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t first = i > bandwidth_ ? i - bandwidth_ : 0;
            for (std::size_t j = first; j < i; ++j) rhs[i] -= lower(i, j) * rhs[j];
            rhs[i] /= lower(i, i);
        }
        for (std::size_t ii = count; ii-- > 0;) {
            const std::size_t last = std::min(count - 1, ii + bandwidth_);
            for (std::size_t j = ii + 1; j <= last; ++j) rhs[ii] -= lower(j, ii) * rhs[j];
            rhs[ii] /= lower(ii, ii);
        }
        for (std::size_t k = 0; k < count; ++k) a[full_index(unknown_nodes_[k])] = rhs[k];
        sync_periodic(a);
    }

    // Native dual-volume current. At a nonperiodic outer boundary, where no
    // complete dual volume exists, use the nearest three-point one-sided
    // second derivative. Those vertices are never in the powered mask.
    std::vector<double> current(const std::vector<double>& a) const {
        check_nodes(a);
        std::vector<double> result(node_count(), 0.0);
        for (int j = 0; j < unique_ny_; ++j) {
            for (int i = 0; i < unique_nx_; ++i) {
                result[node_index(i, j)] = -second_x(a, i, j) - second_y(a, i, j);
            }
        }
        sync_periodic(result);
        return result;
    }

    std::vector<double> residual(const std::vector<double>& a,
                                 const std::vector<double>& target) const {
        check_nodes(a);
        check_nodes(target);
        std::vector<double> result(node_count(), 0.0);
        for (const std::size_t node : unknown_nodes_) {
            const std::size_t full = full_index(node);
            result[full] = native_current(a, node) - target[full];
        }
        sync_periodic(result);
        return result;
    }

    double max_residual(const std::vector<double>& a,
                        const std::vector<double>& target) const {
        check_nodes(a);
        check_nodes(target);
        double result = 0.0;
        for (const std::size_t node : unknown_nodes_) {
            const double error = std::abs(native_current(a, node) - target[full_index(node)]);
            if (!std::isfinite(error)) return std::numeric_limits<double>::infinity();
            result = std::max(result, error);
        }
        return result;
    }

    // Normal-face arrays have sizes (nx+1)*ny and nx*(ny+1), respectively.
    void face_fields(const std::vector<double>& a, std::vector<double>& bx_face,
                     std::vector<double>& by_face) const {
        check_nodes(a);
        bx_face.resize(static_cast<std::size_t>(nx_ + 1) * ny_);
        by_face.resize(static_cast<std::size_t>(nx_) * (ny_ + 1));
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i <= nx_; ++i) {
                bx_face[static_cast<std::size_t>(j) * (nx_ + 1) + i] =
                    (value(a, i, j + 1) - value(a, i, j)) / dy_;
            }
        }
        for (int j = 0; j <= ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                by_face[static_cast<std::size_t>(j) * nx_ + i] =
                    -(value(a, i + 1, j) - value(a, i, j)) / dx_;
            }
        }
    }

    void cell_fields(const std::vector<double>& a, std::vector<double>& bx_cell,
                     std::vector<double>& by_cell) const {
        check_nodes(a);
        bx_cell.resize(cell_count());
        by_cell.resize(cell_count());
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                const double a00 = value(a, i, j);
                const double a10 = value(a, i + 1, j);
                const double a01 = value(a, i, j + 1);
                const double a11 = value(a, i + 1, j + 1);
                bx_cell[cell_index(i, j)] = (a01 + a11 - a00 - a10) / (2.0 * dy_);
                by_cell[cell_index(i, j)] = -(a10 + a11 - a00 - a01) / (2.0 * dx_);
            }
        }
    }

    // Exact graph magnetic energy corresponding to the native curl/current
    // pair. It averages face B^2, rather than squaring averaged cell B. The
    // two energies differ by a nonnegative, resolution-dependent variance.
    double magnetic_energy(const std::vector<double>& a) const {
        check_nodes(a);
        double sum = 0.0;
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                const double a00 = value(a, i, j);
                const double a10 = value(a, i + 1, j);
                const double a01 = value(a, i, j + 1);
                const double a11 = value(a, i + 1, j + 1);
                const double bxl = (a01 - a00) / dy_;
                const double bxr = (a11 - a10) / dy_;
                const double byb = -(a10 - a00) / dx_;
                const double byt = -(a11 - a01) / dx_;
                sum += bxl * bxl + bxr * bxr + byb * byb + byt * byt;
            }
        }
        return 0.25 * dx_ * dy_ * sum;
    }

    double integrated_current(const std::vector<double>& nodal_current,
                              bool powered_only = true) const {
        check_nodes(nodal_current);
        double sum = 0.0;
        for (std::size_t node = 0; node < unique_node_count(); ++node) {
            if (!powered_only || canonical_powered_[node]) {
                sum += nodal_current[full_index(node)];
            }
        }
        return dx_ * dy_ * sum;
    }

    // Electrical feed power -integral Ja*Ez = integral Ja*A_dot. This is the
    // contribution of the powered vertices to the graph-energy identity;
    // plasma work and outer-boundary flux must also be included in a budget.
    double source_power(const std::vector<double>& a_dot,
                        const std::vector<double>& nodal_current) const {
        check_nodes(a_dot);
        check_nodes(nodal_current);
        double sum = 0.0;
        for (const std::size_t node : unknown_nodes_) {
            const std::size_t full = full_index(node);
            sum += nodal_current[full] * a_dot[full];
        }
        return dx_ * dy_ * sum;
    }

    void sync_periodic(std::vector<double>& a) const {
        check_nodes(a);
        if (periodic_x_) {
            for (int j = 0; j <= ny_; ++j) a[node_index(nx_, j)] = a[node_index(0, j)];
        }
        if (periodic_y_) {
            for (int i = 0; i <= nx_; ++i) a[node_index(i, ny_)] = a[node_index(i, 0)];
        }
    }

private:
    int nx_, ny_;
    double dx_, dy_;
    bool periodic_x_, periodic_y_;
    int unique_nx_, unique_ny_;
    double cx_ = 0.0, cy_ = 0.0;
    std::vector<std::uint8_t> canonical_powered_, powered_mask_;
    std::vector<std::size_t> unknown_nodes_;
    std::vector<int> unknown_index_;
    std::size_t bandwidth_ = 0;
    std::vector<double> cholesky_;

    std::size_t unique_node_count() const {
        return static_cast<std::size_t>(unique_nx_) * unique_ny_;
    }
    std::size_t canonical_index(int i, int j) const {
        if (periodic_x_) {
            i %= nx_;
            if (i < 0) i += nx_;
        }
        if (periodic_y_) {
            j %= ny_;
            if (j < 0) j += ny_;
        }
        return static_cast<std::size_t>(j) * unique_nx_ + static_cast<std::size_t>(i);
    }
    std::size_t full_index(std::size_t canonical) const {
        return node_index(static_cast<int>(canonical % unique_nx_),
                          static_cast<int>(canonical / unique_nx_));
    }
    std::array<std::size_t, 4> neighbor_nodes(std::size_t node) const {
        const int i = static_cast<int>(node % unique_nx_);
        const int j = static_cast<int>(node / unique_nx_);
        return {canonical_index(i - 1, j), canonical_index(i + 1, j),
                canonical_index(i, j - 1), canonical_index(i, j + 1)};
    }
    double coefficient(std::size_t direction) const { return direction < 2 ? cx_ : cy_; }
    double value(const std::vector<double>& a, int i, int j) const {
        return a[full_index(canonical_index(i, j))];
    }
    double native_current(const std::vector<double>& a, std::size_t node) const {
        const auto neighbors = neighbor_nodes(node);
        double result = 2.0 * (cx_ + cy_) * a[full_index(node)];
        for (std::size_t d = 0; d < 4; ++d) {
            result -= coefficient(d) * a[full_index(neighbors[d])];
        }
        return result;
    }
    double second_x(const std::vector<double>& a, int i, int j) const {
        if (!periodic_x_ && i == 0) {
            return (value(a, 0, j) - 2.0 * value(a, 1, j) + value(a, 2, j)) * cx_;
        }
        if (!periodic_x_ && i == nx_) {
            return (value(a, nx_, j) - 2.0 * value(a, nx_ - 1, j)
                    + value(a, nx_ - 2, j)) * cx_;
        }
        return (value(a, i - 1, j) - 2.0 * value(a, i, j) + value(a, i + 1, j)) * cx_;
    }
    double second_y(const std::vector<double>& a, int i, int j) const {
        if (!periodic_y_ && j == 0) {
            return (value(a, i, 0) - 2.0 * value(a, i, 1) + value(a, i, 2)) * cy_;
        }
        if (!periodic_y_ && j == ny_) {
            return (value(a, i, ny_) - 2.0 * value(a, i, ny_ - 1)
                    + value(a, i, ny_ - 2)) * cy_;
        }
        return (value(a, i, j - 1) - 2.0 * value(a, i, j) + value(a, i, j + 1)) * cy_;
    }
    void check_nodes(const std::vector<double>& a) const {
        if (a.size() != node_count()) {
            throw std::invalid_argument("Prescribed-current nodal array has wrong size");
        }
    }
    void check_cells(const std::vector<double>& a) const {
        if (a.size() != cell_count()) {
            throw std::invalid_argument("Prescribed-current cell array has wrong size");
        }
    }
    double& lower(std::size_t i, std::size_t j) {
        return cholesky_[i * (bandwidth_ + 1) + i - j];
    }
    double lower(std::size_t i, std::size_t j) const {
        return cholesky_[i * (bandwidth_ + 1) + i - j];
    }

    void build_factorization() {
        std::vector<std::size_t> natural;
        for (std::size_t node = 0; node < unique_node_count(); ++node) {
            if (canonical_powered_[node]) natural.push_back(node);
        }
        unknown_index_.assign(unique_node_count(), -1);
        if (natural.empty()) return;
        if (periodic_x_ && periodic_y_ && natural.size() == unique_node_count()) {
            throw std::invalid_argument(
                "A fully periodic powered-current domain requires a free plasma vertex");
        }

        // Reverse Cuthill-McKee reduces the band for oblique/thin bodies and
        // periodic slabs. Retain natural ordering if its band is smaller.
        std::vector<std::vector<std::size_t>> adjacency(unique_node_count());
        for (const std::size_t node : natural) {
            auto& row = adjacency[node];
            for (const std::size_t neighbor : neighbor_nodes(node)) {
                if (canonical_powered_[neighbor]) row.push_back(neighbor);
            }
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
        }
        std::vector<std::uint8_t> seen(unique_node_count(), 0);
        std::vector<std::size_t> reordered;
        reordered.reserve(natural.size());
        while (reordered.size() < natural.size()) {
            std::size_t start = natural.front();
            std::size_t degree = std::numeric_limits<std::size_t>::max();
            for (const std::size_t node : natural) {
                if (!seen[node] && adjacency[node].size() < degree) {
                    start = node;
                    degree = adjacency[node].size();
                }
            }
            std::vector<std::size_t> queue{start};
            seen[start] = 1;
            for (std::size_t head = 0; head < queue.size(); ++head) {
                std::vector<std::size_t> next;
                for (const std::size_t neighbor : adjacency[queue[head]]) {
                    if (!seen[neighbor]) {
                        seen[neighbor] = 1;
                        next.push_back(neighbor);
                    }
                }
                std::sort(next.begin(), next.end(), [&](std::size_t a, std::size_t b) {
                    if (adjacency[a].size() != adjacency[b].size()) {
                        return adjacency[a].size() < adjacency[b].size();
                    }
                    return a < b;
                });
                queue.insert(queue.end(), next.begin(), next.end());
            }
            reordered.insert(reordered.end(), queue.rbegin(), queue.rend());
        }
        auto band_for = [&](const std::vector<std::size_t>& order) {
            for (std::size_t k = 0; k < order.size(); ++k) {
                unknown_index_[order[k]] = static_cast<int>(k);
            }
            std::size_t band = 0;
            for (std::size_t k = 0; k < order.size(); ++k) {
                for (const std::size_t neighbor : adjacency[order[k]]) {
                    const std::size_t other = static_cast<std::size_t>(unknown_index_[neighbor]);
                    band = std::max(band, k > other ? k - other : other - k);
                }
            }
            return band;
        };
        const std::size_t natural_band = band_for(natural);
        const std::size_t reordered_band = band_for(reordered);
        unknown_nodes_ = reordered_band < natural_band ? std::move(reordered) : std::move(natural);
        bandwidth_ = std::min(natural_band, reordered_band);
        for (std::size_t k = 0; k < unknown_nodes_.size(); ++k) {
            unknown_index_[unknown_nodes_[k]] = static_cast<int>(k);
        }
        const std::size_t count = unknown_nodes_.size();
        cholesky_.assign(count * (bandwidth_ + 1), 0.0);
        for (std::size_t row = 0; row < count; ++row) {
            lower(row, row) = 2.0 * (cx_ + cy_);
            const auto neighbors = neighbor_nodes(unknown_nodes_[row]);
            for (std::size_t d = 0; d < 4; ++d) {
                const int column = unknown_index_[neighbors[d]];
                if (column >= 0 && static_cast<std::size_t>(column) <= row) {
                    lower(row, static_cast<std::size_t>(column)) -= coefficient(d);
                }
            }
        }
        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t first = i > bandwidth_ ? i - bandwidth_ : 0;
            for (std::size_t j = first; j <= i; ++j) {
                double sum = lower(i, j);
                for (std::size_t k = first; k < j; ++k) sum -= lower(i, k) * lower(j, k);
                if (i == j) {
                    if (!(sum > 0.0) || !std::isfinite(sum)) {
                        throw std::runtime_error(
                            "Powered-current Poisson matrix is not positive definite; "
                            "a fully periodic domain requires at least one free plasma vertex");
                    }
                    lower(i, i) = std::sqrt(sum);
                } else {
                    lower(i, j) = sum / lower(j, j);
                }
            }
        }
    }
};

#endif

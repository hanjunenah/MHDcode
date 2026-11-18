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

#include "prescribed_current.hpp"

namespace {

constexpr std::size_t nvar = 9;
enum Variable : std::size_t { rho, mx, my, mz, bx, by, bz, total_energy, psi };
using State = std::array<double, nvar>;

struct Config {
    int nx = 0;
    int ny = 0;
    double lx = 0.0;
    double ly = 0.0;
    double gamma = 5.0 / 3.0;
    double cfl = 0.35;
    double diffusion_cfl = 0.45;
    double resistivity = 0.0;
    std::string antenna_model = "resistive_impressed";
    double t_end = 0.0;
    int max_steps = 0;
    bool second_order = true;
    double limiter_theta = 1.5;
    double glm_speed_factor = 1.0;
    double glm_damping = 0.0;
    double rho_floor = 1.0e-8;
    double p_floor = 1.0e-8;
    double output_interval = 0.1;
    int requested_threads = 0;
    std::string boundary_x = "periodic";
    std::string boundary_y = "periodic";
    double absorption_width = 0.0;
    double absorption_rate = 0.5;
    double absorption_magnetic_diffusivity = 0.05;
    std::array<double, 8> ambient = {1.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

struct InitialData {
    std::vector<double> rho, u, v, w, p, bx, by, bz;
};

struct AntennaData {
    std::vector<double> time, jx, jy, jz;

    bool enabled() const { return !time.empty(); }
    std::size_t sample_count() const { return time.size(); }
};

struct SolverInput {
    Config config;
    InitialData initial;
    AntennaData antenna;
    std::vector<double> solid_mask;
};

std::string format_wall_time(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) return "--";
    const auto total_seconds = static_cast<long long>(seconds);
    const long long hours = total_seconds / 3600;
    const long long minutes = (total_seconds % 3600) / 60;
    const double remaining_seconds = seconds - static_cast<double>(hours * 3600 + minutes * 60);
    std::ostringstream output;
    output << std::setfill('0');
    if (hours > 0) {
        output << hours << ':' << std::setw(2) << minutes << ':'
               << std::fixed << std::setprecision(1) << std::setw(4) << remaining_seconds;
    } else if (minutes > 0) {
        output << minutes << ':' << std::fixed << std::setprecision(1)
               << std::setw(4) << remaining_seconds;
    } else {
        output << std::fixed << std::setprecision(2) << remaining_seconds << " s";
    }
    return output.str();
}

void h5_check(herr_t status, const std::string& operation) {
    if (status < 0) throw std::runtime_error("HDF5 error while " + operation);
}

hid_t h5_id(hid_t identifier, const std::string& operation) {
    if (identifier < 0) throw std::runtime_error("HDF5 error while " + operation);
    return identifier;
}

template <typename T> hid_t h5_native_type();
template <> hid_t h5_native_type<int>() { return H5T_NATIVE_INT; }
template <> hid_t h5_native_type<double>() { return H5T_NATIVE_DOUBLE; }

template <typename T>
T read_attribute(hid_t object, const char* name) {
    const hid_t attribute = h5_id(H5Aopen(object, name, H5P_DEFAULT), std::string("opening attribute ") + name);
    T value{};
    const herr_t status = H5Aread(attribute, h5_native_type<T>(), &value);
    H5Aclose(attribute);
    h5_check(status, std::string("reading attribute ") + name);
    return value;
}

template <typename T>
T read_optional_attribute(hid_t object, const char* name, T fallback) {
    const htri_t exists = H5Aexists(object, name);
    h5_check(exists, std::string("checking attribute ") + name);
    return exists > 0 ? read_attribute<T>(object, name) : fallback;
}

std::string read_optional_string_attribute(hid_t object, const char* name,
                                           const std::string& fallback) {
    const htri_t exists = H5Aexists(object, name);
    h5_check(exists, std::string("checking attribute ") + name);
    if (exists == 0) return fallback;
    const hid_t attribute = h5_id(H5Aopen(object, name, H5P_DEFAULT),
                                  std::string("opening attribute ") + name);
    const hid_t type = h5_id(H5Aget_type(attribute), "getting string attribute type");
    if (H5Tget_class(type) != H5T_STRING) {
        H5Tclose(type); H5Aclose(attribute);
        throw std::runtime_error(std::string(name) + " must be a string");
    }
    std::string value;
    herr_t status = 0;
    if (H5Tis_variable_str(type) > 0) {
        char* buffer = nullptr;
        status = H5Aread(attribute, type, &buffer);
        if (status >= 0 && buffer != nullptr) value = buffer;
        if (buffer != nullptr) H5free_memory(buffer);
    } else {
        std::vector<char> buffer(H5Tget_size(type) + 1, '\0');
        status = H5Aread(attribute, type, buffer.data());
        if (status >= 0) value = buffer.data();
    }
    H5Tclose(type); H5Aclose(attribute);
    h5_check(status, std::string("reading attribute ") + name);
    return value;
}

template <typename T>
void write_attribute(hid_t object, const char* name, T value) {
    const hid_t space = h5_id(H5Screate(H5S_SCALAR), std::string("creating attribute space for ") + name);
    const hid_t attribute = h5_id(
        H5Acreate2(object, name, h5_native_type<T>(), space, H5P_DEFAULT, H5P_DEFAULT),
        std::string("creating attribute ") + name);
    const herr_t status = H5Awrite(attribute, h5_native_type<T>(), &value);
    H5Aclose(attribute);
    H5Sclose(space);
    h5_check(status, std::string("writing attribute ") + name);
}

void write_string_attribute(hid_t object, const char* name, const std::string& value) {
    const hid_t type = h5_id(H5Tcopy(H5T_C_S1), "copying string type");
    h5_check(H5Tset_size(type, value.size() + 1), "sizing string attribute");
    h5_check(H5Tset_strpad(type, H5T_STR_NULLTERM), "setting string padding");
    const hid_t space = h5_id(H5Screate(H5S_SCALAR), "creating string attribute space");
    const hid_t attribute = h5_id(
        H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT),
        std::string("creating attribute ") + name);
    const herr_t status = H5Awrite(attribute, type, value.c_str());
    H5Aclose(attribute);
    H5Sclose(space);
    H5Tclose(type);
    h5_check(status, std::string("writing attribute ") + name);
}

std::vector<double> read_2d_dataset(hid_t file, const std::string& name, int ny, int nx) {
    const hid_t dataset = h5_id(H5Dopen2(file, name.c_str(), H5P_DEFAULT), "opening " + name);
    const hid_t space = h5_id(H5Dget_space(dataset), "getting dimensions for " + name);
    if (H5Sget_simple_extent_ndims(space) != 2) {
        H5Sclose(space); H5Dclose(dataset);
        throw std::runtime_error(name + " must be a two-dimensional dataset");
    }
    hsize_t dimensions[2]{};
    H5Sget_simple_extent_dims(space, dimensions, nullptr);
    if (dimensions[0] != static_cast<hsize_t>(ny) || dimensions[1] != static_cast<hsize_t>(nx)) {
        H5Sclose(space); H5Dclose(dataset);
        throw std::runtime_error(name + " has the wrong grid shape");
    }
    std::vector<double> values(static_cast<std::size_t>(nx * ny));
    const herr_t status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
    H5Sclose(space);
    H5Dclose(dataset);
    h5_check(status, "reading " + name);
    return values;
}

std::vector<double> read_1d_dataset(hid_t file, const std::string& name) {
    const hid_t dataset = h5_id(H5Dopen2(file, name.c_str(), H5P_DEFAULT), "opening " + name);
    const hid_t space = h5_id(H5Dget_space(dataset), "getting dimensions for " + name);
    if (H5Sget_simple_extent_ndims(space) != 1) {
        H5Sclose(space); H5Dclose(dataset);
        throw std::runtime_error(name + " must be a one-dimensional dataset");
    }
    hsize_t dimension{};
    H5Sget_simple_extent_dims(space, &dimension, nullptr);
    std::vector<double> values(static_cast<std::size_t>(dimension));
    const herr_t status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                                  H5P_DEFAULT, values.data());
    H5Sclose(space); H5Dclose(dataset);
    h5_check(status, "reading " + name);
    return values;
}

std::vector<double> read_3d_dataset(hid_t file, const std::string& name,
                                    std::size_t nt, int ny, int nx) {
    const hid_t dataset = h5_id(H5Dopen2(file, name.c_str(), H5P_DEFAULT), "opening " + name);
    const hid_t space = h5_id(H5Dget_space(dataset), "getting dimensions for " + name);
    if (H5Sget_simple_extent_ndims(space) != 3) {
        H5Sclose(space); H5Dclose(dataset);
        throw std::runtime_error(name + " must be a three-dimensional dataset");
    }
    hsize_t dimensions[3]{};
    H5Sget_simple_extent_dims(space, dimensions, nullptr);
    if (dimensions[0] != static_cast<hsize_t>(nt) ||
        dimensions[1] != static_cast<hsize_t>(ny) ||
        dimensions[2] != static_cast<hsize_t>(nx)) {
        H5Sclose(space); H5Dclose(dataset);
        throw std::runtime_error(name + " has the wrong antenna-table shape");
    }
    std::vector<double> values(nt * static_cast<std::size_t>(nx * ny));
    const herr_t status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                                  H5P_DEFAULT, values.data());
    H5Sclose(space); H5Dclose(dataset);
    h5_check(status, "reading " + name);
    return values;
}

SolverInput read_solver_input(const std::string& filename) {
    const hid_t file = h5_id(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT),
                             "opening solver input " + filename);
    const hid_t group = h5_id(H5Gopen2(file, "/config", H5P_DEFAULT), "opening /config");
    Config c;
    c.nx = read_attribute<int>(group, "nx");
    c.ny = read_attribute<int>(group, "ny");
    c.lx = read_attribute<double>(group, "lx");
    c.ly = read_attribute<double>(group, "ly");
    c.gamma = read_attribute<double>(group, "gamma");
    c.cfl = read_attribute<double>(group, "cfl");
    c.diffusion_cfl = read_attribute<double>(group, "diffusion_cfl");
    c.resistivity = read_attribute<double>(group, "resistivity");
    c.antenna_model = read_optional_string_attribute(group, "antenna_model", "resistive_impressed");
    c.t_end = read_attribute<double>(group, "t_end");
    c.max_steps = read_attribute<int>(group, "max_steps");
    c.second_order = read_attribute<int>(group, "second_order") != 0;
    c.limiter_theta = read_attribute<double>(group, "limiter_theta");
    c.glm_speed_factor = read_attribute<double>(group, "glm_speed_factor");
    c.glm_damping = read_attribute<double>(group, "glm_damping");
    c.rho_floor = read_attribute<double>(group, "rho_floor");
    c.p_floor = read_attribute<double>(group, "p_floor");
    c.output_interval = read_attribute<double>(group, "output_interval");
    c.requested_threads = read_attribute<int>(group, "threads");
    c.boundary_x = read_optional_string_attribute(group, "boundary_x", "periodic");
    c.boundary_y = read_optional_string_attribute(group, "boundary_y", "periodic");
    c.absorption_width = read_optional_attribute<double>(group, "absorption_width", 0.0);
    c.absorption_rate = read_optional_attribute<double>(group, "absorption_rate", 0.5);
    c.absorption_magnetic_diffusivity = read_optional_attribute<double>(group, "absorption_magnetic_diffusivity", 0.05);
    H5Gclose(group);

    const bool absorbing = c.boundary_x == "absorbing" || c.boundary_y == "absorbing";
    const htri_t boundary_group_exists = H5Lexists(file, "/boundary", H5P_DEFAULT);
    h5_check(boundary_group_exists, "checking for /boundary");
    const htri_t ambient_exists = boundary_group_exists > 0 ? H5Lexists(file, "/boundary/ambient", H5P_DEFAULT) : 0;
    h5_check(ambient_exists, "checking for /boundary/ambient");
    if (absorbing && ambient_exists == 0) {
        throw std::runtime_error("Absorbing boundaries require an explicit /boundary/ambient state");
    }
    if (ambient_exists > 0) {
        const hid_t ambient_group = h5_id(H5Gopen2(file, "/boundary/ambient", H5P_DEFAULT), "opening ambient state");
        const std::array<const char*, 8> names = {"rho", "p", "u", "v", "w", "Bx", "By", "Bz"};
        for (std::size_t k = 0; k < names.size(); ++k) c.ambient[k] = read_attribute<double>(ambient_group, names[k]);
        H5Gclose(ambient_group);
    }

    InitialData initial;
    initial.rho = read_2d_dataset(file, "/initial/rho", c.ny, c.nx);
    initial.u = read_2d_dataset(file, "/initial/u", c.ny, c.nx);
    initial.v = read_2d_dataset(file, "/initial/v", c.ny, c.nx);
    initial.w = read_2d_dataset(file, "/initial/w", c.ny, c.nx);
    initial.p = read_2d_dataset(file, "/initial/p", c.ny, c.nx);
    initial.bx = read_2d_dataset(file, "/initial/Bx", c.ny, c.nx);
    initial.by = read_2d_dataset(file, "/initial/By", c.ny, c.nx);
    initial.bz = read_2d_dataset(file, "/initial/Bz", c.ny, c.nx);

    AntennaData antenna;
    const htri_t antenna_exists = H5Lexists(file, "/antenna", H5P_DEFAULT);
    h5_check(antenna_exists, "checking for /antenna");
    if (antenna_exists > 0) {
        antenna.time = read_1d_dataset(file, "/antenna/time");
        if (antenna.time.empty()) throw std::runtime_error("/antenna/time must not be empty");
        antenna.jx = read_3d_dataset(file, "/antenna/Jx", antenna.time.size(), c.ny, c.nx);
        antenna.jy = read_3d_dataset(file, "/antenna/Jy", antenna.time.size(), c.ny, c.nx);
        antenna.jz = read_3d_dataset(file, "/antenna/Jz", antenna.time.size(), c.ny, c.nx);
    }
    std::vector<double> solid_mask;
    const htri_t solid_group_exists = H5Lexists(file, "/solid", H5P_DEFAULT);
    h5_check(solid_group_exists, "checking for /solid");
    const htri_t solid_exists = solid_group_exists > 0 ? H5Lexists(file, "/solid/mask", H5P_DEFAULT) : 0;
    h5_check(solid_exists, "checking for /solid/mask");
    if (solid_exists > 0) {
        solid_mask = read_2d_dataset(file, "/solid/mask", c.ny, c.nx);
        bool any_solid = false, any_fluid = false;
        for (std::size_t cell = 0; cell < solid_mask.size(); ++cell) {
            if (solid_mask[cell] != 0.0 && solid_mask[cell] != 1.0) {
                throw std::runtime_error("solid mask values must be exactly zero or one");
            }
            if (solid_mask[cell] == 1.0) {
                any_solid = true;
                if (initial.u[cell] != 0.0 || initial.v[cell] != 0.0 || initial.w[cell] != 0.0) {
                    throw std::runtime_error("Initial velocity inside a stationary solid must be zero");
                }
            } else {
                any_fluid = true;
            }
        }
        if (!any_fluid) throw std::runtime_error("The solid mask must leave at least one fluid cell");
        if (!any_solid) solid_mask.clear();
    }
    H5Fclose(file);

    if (c.nx < 4 || c.ny < 4 || c.lx <= 0.0 || c.ly <= 0.0) {
        throw std::runtime_error("The grid must have at least 4 cells per direction and positive lengths");
    }
    if ((c.boundary_x != "periodic" && c.boundary_x != "outflow" && c.boundary_x != "absorbing") ||
        (c.boundary_y != "periodic" && c.boundary_y != "outflow" && c.boundary_y != "absorbing")) {
        throw std::runtime_error("boundary_x and boundary_y must be periodic, outflow, or absorbing");
    }
    if (!std::isfinite(c.absorption_width) || !std::isfinite(c.absorption_rate) ||
        !std::isfinite(c.absorption_magnetic_diffusivity) || c.absorption_width < 0.0 ||
        c.absorption_rate < 0.0 || c.absorption_magnetic_diffusivity < 0.0) {
        throw std::runtime_error("Absorption width, rate and magnetic diffusivity must be finite and non-negative");
    }
    if (absorbing && (c.absorption_width <= 0.0 || c.absorption_rate <= 0.0 ||
        (c.boundary_x == "absorbing" && 2.0 * c.absorption_width >= c.lx) ||
        (c.boundary_y == "absorbing" && 2.0 * c.absorption_width >= c.ly))) {
        throw std::runtime_error("Absorbing boundaries require positive width/rate and width below half the absorbing domain extent");
    }
    if ((absorbing || ambient_exists > 0) &&
        (!std::all_of(c.ambient.begin(), c.ambient.end(), [](double v) { return std::isfinite(v); }) ||
         c.ambient[0] < c.rho_floor || c.ambient[1] < c.p_floor)) {
        throw std::runtime_error("Ambient state must be finite with density/pressure above the floors");
    }
    if (c.gamma <= 1.0 || c.cfl <= 0.0 || c.cfl >= 1.0 || c.diffusion_cfl <= 0.0 ||
        c.diffusion_cfl >= 0.5 || c.resistivity < 0.0) {
        throw std::runtime_error("Invalid thermodynamic, CFL, or resistivity parameter");
    }
    if (c.t_end < 0.0 || c.max_steps <= 0 || c.output_interval <= 0.0) {
        throw std::runtime_error("Invalid run duration or output interval");
    }
    if (c.limiter_theta < 1.0 || c.limiter_theta > 2.0 || c.glm_speed_factor <= 0.0 ||
        c.glm_damping < 0.0 || c.rho_floor <= 0.0 || c.p_floor <= 0.0 ||
        c.requested_threads < 0) {
        throw std::runtime_error("Invalid limiter, GLM, floor, or thread setting");
    }
    if (antenna.enabled()) {
        if (antenna.time.front() != 0.0 || antenna.time.back() < c.t_end ||
            !std::all_of(antenna.time.begin(), antenna.time.end(),
                         [](double value) { return std::isfinite(value); })) {
            throw std::runtime_error("Antenna times must be finite and cover 0 <= t <= t_end");
        }
        for (std::size_t sample = 1; sample < antenna.time.size(); ++sample) {
            if (!(antenna.time[sample] > antenna.time[sample - 1])) {
                throw std::runtime_error("Antenna sample times must be strictly increasing");
            }
        }
        auto finite_table = [](const std::vector<double>& values) {
            return std::all_of(values.begin(), values.end(),
                               [](double value) { return std::isfinite(value); });
        };
        if (!finite_table(antenna.jx) || !finite_table(antenna.jy) || !finite_table(antenna.jz)) {
            throw std::runtime_error("Antenna current table contains a non-finite value");
        }
    }
    return {c, std::move(initial), std::move(antenna), std::move(solid_mask)};
}

class ParallelExecutor {
public:
    explicit ParallelExecutor(int requested_threads) {
        const unsigned int available = std::max(1U, std::thread::hardware_concurrency());
        thread_count_ = requested_threads == 0 ? static_cast<int>(available) : requested_threads;
        thread_count_ = std::max(1, thread_count_);
        for (int i = 0; i < thread_count_; ++i) workers_.emplace_back([this, i] { worker(i); });
    }

    ParallelExecutor(const ParallelExecutor&) = delete;
    ParallelExecutor& operator=(const ParallelExecutor&) = delete;

    ~ParallelExecutor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        work_available_.notify_all();
        for (std::thread& thread : workers_) thread.join();
    }

    int thread_count() const { return thread_count_; }

    template <typename Function>
    void for_each(std::size_t count, Function&& function) {
        if (count == 0) return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            task_ = std::forward<Function>(function);
            count_ = count;
            completed_workers_ = 0;
            exception_ = nullptr;
            ++generation_;
        }
        work_available_.notify_all();
        std::unique_lock<std::mutex> lock(mutex_);
        work_complete_.wait(lock, [this] { return completed_workers_ == workers_.size(); });
        if (exception_) std::rethrow_exception(exception_);
    }

private:
    int thread_count_ = 1;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable work_available_;
    std::condition_variable work_complete_;
    std::function<void(std::size_t)> task_;
    std::size_t count_ = 0;
    std::size_t generation_ = 0;
    std::size_t completed_workers_ = 0;
    bool stopping_ = false;
    std::exception_ptr exception_;

    void worker(int worker_index) {
        std::size_t observed_generation = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            work_available_.wait(lock, [this, observed_generation] {
                return stopping_ || generation_ != observed_generation;
            });
            if (stopping_) return;
            observed_generation = generation_;
            const auto task = task_;
            const std::size_t count = count_;
            const std::size_t begin = count * static_cast<std::size_t>(worker_index) /
                                      static_cast<std::size_t>(thread_count_);
            const std::size_t end = count * static_cast<std::size_t>(worker_index + 1) /
                                    static_cast<std::size_t>(thread_count_);
            lock.unlock();
            try {
                for (std::size_t item = begin; item < end; ++item) task(item);
            } catch (...) {
                std::lock_guard<std::mutex> exception_lock(mutex_);
                if (!exception_) exception_ = std::current_exception();
            }
            lock.lock();
            ++completed_workers_;
            if (completed_workers_ == workers_.size()) work_complete_.notify_one();
        }
    }
};

class Hdf5Output {
public:
    Hdf5Output(const std::string& filename, const Config& c, const AntennaData& antenna,
               int threads_used, const std::vector<double>& solid_mask)
        : nx_(c.nx), ny_(c.ny) {
        const std::filesystem::path path(filename);
        if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
        file_ = h5_id(H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
                      "creating output file " + filename);
        write_attribute<int>(file_, "format_version", 3);
        write_string_attribute(file_, "model", c.antenna_model == "prescribed_current" ?
            "dimensionless planar resistive CT-MHD with powered prescribed-current bodies" :
            "dimensionless 2.5-D driven resistive GLM-MHD");
        write_attribute<int>(file_, "nx", c.nx); write_attribute<int>(file_, "ny", c.ny);
        write_attribute<double>(file_, "lx", c.lx); write_attribute<double>(file_, "ly", c.ly);
        write_attribute<double>(file_, "gamma", c.gamma);
        write_attribute<double>(file_, "resistivity", c.resistivity);
        write_string_attribute(file_, "antenna_model", c.antenna_model);
        write_string_attribute(file_, "divB_definition", c.antenna_model == "prescribed_current" ?
            "finite-volume divergence of native CT normal face fields" :
            "centered derivative of cell-centered magnetic fields");
        write_string_attribute(file_, "boundary_x", c.boundary_x);
        write_string_attribute(file_, "boundary_y", c.boundary_y);
        write_attribute<double>(file_, "absorption_width", c.absorption_width);
        write_attribute<double>(file_, "absorption_rate", c.absorption_rate);
        write_attribute<double>(file_, "absorption_magnetic_diffusivity", c.absorption_magnetic_diffusivity);
        if (c.boundary_x == "absorbing" || c.boundary_y == "absorbing") {
            write_string_attribute(file_, "absorption_model",
                "ambient HLLD far-field flux plus smooth fluid relaxation and divergence-preserving curl-form magnetic diffusion sponge; not a perfectly matched layer");
            write_string_attribute(file_, "absorption_energy_definition",
                "signed energy supplied to the computational domain by the artificial absorber; magnetic damping preserves local thermal pressure");
            const hid_t boundary_group = h5_id(H5Gcreate2(file_, "boundary", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "creating boundary group");
            const hid_t ambient_group = h5_id(H5Gcreate2(boundary_group, "ambient", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "creating ambient group");
            const std::array<const char*, 8> names = {"rho", "p", "u", "v", "w", "Bx", "By", "Bz"};
            for (std::size_t k = 0; k < names.size(); ++k) write_attribute<double>(ambient_group, names[k], c.ambient[k]);
            H5Gclose(ambient_group); H5Gclose(boundary_group);
        }
        write_attribute<int>(file_, "threads_used", threads_used);
        write_attribute<int>(file_, "antenna_enabled", antenna.enabled() ? 1 : 0);
        const double antenna_sample_interval = antenna.sample_count() > 1
            ? antenna.time[1] - antenna.time[0] : 0.0;
        write_attribute<double>(file_, "antenna_sample_interval", antenna_sample_interval);
        write_attribute<int>(file_, "solid_enabled", solid_mask.empty() ? 0 : 1);
        if (!solid_mask.empty()) {
            write_string_attribute(file_, "solid_model",
                c.antenna_model == "prescribed_current" ?
                "stationary impermeable powered body; prescribed native current; nonuniform Az and penetrating magnetic field" :
                "stationary impermeable slip body; magnetic field penetrates with uniform finite resistivity; not a perfect conductor");
            write_string_attribute(file_, "solid_energy_accounting",
                "body rho and thermal pressure are fixed placeholders; solid_constraint_energy is energy supplied by the constraint reservoir");
            write_string_attribute(file_, "solid_force_definition",
                "last RK2 step-average force of plasma and magnetic traction on the fixed body; support reaction is its negative");
            const hid_t group = h5_id(H5Gcreate2(file_, "solid", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                                       "creating solid group");
            const hsize_t dimensions[2] = {static_cast<hsize_t>(ny_), static_cast<hsize_t>(nx_)};
            const hid_t space = h5_id(H5Screate_simple(2, dimensions, nullptr), "creating solid mask space");
            const hid_t dataset = h5_id(H5Dcreate2(group, "mask", H5T_IEEE_F64LE, space,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "creating solid mask");
            h5_check(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                              solid_mask.data()), "writing solid mask");
            H5Dclose(dataset); H5Sclose(space); H5Gclose(group);
        }

        std::vector<double> x(static_cast<std::size_t>(nx_)), y(static_cast<std::size_t>(ny_));
        for (int i = 0; i < nx_; ++i) x[static_cast<std::size_t>(i)] = (i + 0.5) * c.lx / nx_;
        for (int j = 0; j < ny_; ++j) y[static_cast<std::size_t>(j)] = (j + 0.5) * c.ly / ny_;
        write_fixed_1d("x", x); write_fixed_1d("y", y);
        time_dataset_ = create_series("time");

        const hid_t fields_group = h5_id(H5Gcreate2(file_, "fields", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                                          "creating fields group");
        for (const std::string& name : field_names()) field_datasets_[name] = create_field(fields_group, name);
        H5Gclose(fields_group);

        const hid_t diagnostics_group = h5_id(
            H5Gcreate2(file_, "diagnostics", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
            "creating diagnostics group");
        for (const std::string& name : diagnostic_names()) {
            diagnostic_datasets_[name] = create_series(name, diagnostics_group);
        }
        H5Gclose(diagnostics_group);
        if (c.antenna_model == "prescribed_current") {
            const hid_t group = h5_id(H5Gcreate2(file_, "prescribed_current", H5P_DEFAULT,
                H5P_DEFAULT, H5P_DEFAULT), "creating prescribed current group");
            write_string_attribute(group, "definition",
                "vertex Az; Bx=d_y Az, By=-d_x Az on faces; Jz=-Lap_h Az on vertex dual volumes; target is conservative quarter-sum of adjacent cell commands");
            write_string_attribute(group, "energy_definition",
                "antenna work is sum Jz_target*Adot*dx*dy; solid_constraint_energy includes the powered-body FV energy reset; current_drive_energy_correction=solid_constraint_energy-antenna_energy_injected; do not add both solid constraint and antenna work in a budget");
            write_string_attribute(group, "energy_correction_definition",
                "source/body spatial-coupling error, distinct from face-versus-cell magnetic averaging; a closed reservoir budget alone does not establish convergence of circuit power");
            write_string_attribute(group, "geometry_definition",
                "powered nodes are the union of solid-cell corners, including zero-target nodes; electromagnetic dual volumes extend half a cell beyond mechanical body");
            write_string_attribute(group, "power_definition",
                "antenna_power is the last RK2 step-average native electrical power; antenna_energy_injected is its time integral");
            const PrescribedCurrentOperator op(c.nx, c.ny, c.lx / c.nx, c.ly / c.ny, solid_mask);
            std::vector<double> mask(op.node_count());
            for (std::size_t n = 0; n < mask.size(); ++n) mask[n] = op.powered_mask()[n];
            const int old_nx = nx_, old_ny = ny_;
            nx_++; ny_++;
            for (const std::string name : {"Az", "Jz", "Jz_target"})
                node_datasets_[name] = create_field(group, name);
            const hsize_t dims[2] = {static_cast<hsize_t>(ny_), static_cast<hsize_t>(nx_)};
            const hid_t space = h5_id(H5Screate_simple(2, dims, nullptr), "creating powered mask space");
            const hid_t dataset = h5_id(H5Dcreate2(group, "powered_mask", H5T_IEEE_F64LE,
                space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "creating powered mask");
            h5_check(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, mask.data()), "writing powered mask");
            H5Dclose(dataset); H5Sclose(space);
            nx_ = old_nx; ny_ = old_ny;
            std::vector<double> xn(c.nx + 1), yn(c.ny + 1);
            for (int i = 0; i <= c.nx; ++i) xn[i] = i * c.lx / c.nx;
            for (int j = 0; j <= c.ny; ++j) yn[j] = j * c.ly / c.ny;
            write_fixed_1d("prescribed_current/x", xn);
            write_fixed_1d("prescribed_current/y", yn);
            const hid_t diagnostic_group = h5_id(H5Gopen2(file_, "diagnostics", H5P_DEFAULT), "opening diagnostics");
            for (const std::string name : {"max_current_constraint_error", "max_stage_current_constraint_error",
                 "max_abs_ct_divB", "native_magnetic_energy", "magnetic_averaging_energy", "current_drive_energy_correction"})
                diagnostic_datasets_[name] = create_series(name, diagnostic_group);
            H5Gclose(diagnostic_group);
            H5Gclose(group);
        }
    }

    Hdf5Output(const Hdf5Output&) = delete;
    Hdf5Output& operator=(const Hdf5Output&) = delete;

    ~Hdf5Output() {
        for (auto& item : field_datasets_) H5Dclose(item.second);
        for (auto& item : diagnostic_datasets_) H5Dclose(item.second);
        for (auto& item : node_datasets_) H5Dclose(item.second);
        if (time_dataset_ >= 0) H5Dclose(time_dataset_);
        if (file_ >= 0) H5Fclose(file_);
    }

    void append(double time, const std::map<std::string, std::vector<double>>& fields,
                const std::map<std::string, double>& diagnostics,
                const std::map<std::string, std::vector<double>>& nodes = {}) {
        append_scalar(time_dataset_, time);
        for (const auto& item : field_datasets_) {
            const auto found = fields.find(item.first);
            if (found == fields.end()) throw std::runtime_error("Missing output field " + item.first);
            append_field(item.second, found->second);
        }
        for (const auto& item : diagnostic_datasets_) {
            const auto found = diagnostics.find(item.first);
            if (found == diagnostics.end()) throw std::runtime_error("Missing diagnostic " + item.first);
            append_scalar(item.second, found->second);
        }
        if (!node_datasets_.empty()) {
            nx_++; ny_++;
            for (const auto& item : node_datasets_) append_field(item.second, nodes.at(item.first));
            nx_--; ny_--;
        }
        ++frame_;
        h5_check(H5Fflush(file_, H5F_SCOPE_LOCAL), "flushing output file");
    }

private:
    hid_t file_ = -1;
    hid_t time_dataset_ = -1;
    int nx_, ny_;
    hsize_t frame_ = 0;
    std::map<std::string, hid_t> field_datasets_;
    std::map<std::string, hid_t> diagnostic_datasets_;
    std::map<std::string, hid_t> node_datasets_;

    static const std::vector<std::string>& field_names() {
        static const std::vector<std::string> names = {
            "rho", "u", "v", "w", "p", "Bx", "By", "Bz", "Jx", "Jy", "Jz",
            "Jx_antenna", "Jy_antenna", "Jz_antenna", "divB", "psi"
        };
        return names;
    }

    static const std::vector<std::string>& diagnostic_names() {
        static const std::vector<std::string> names = {
            "step", "dt", "mass", "momentum_x", "momentum_y", "momentum_z",
            "magnetic_flux_x", "magnetic_flux_y", "magnetic_flux_z", "glm_integral",
            "entropy_integral", "total_energy", "magnetic_energy", "kinetic_energy", "thermal_energy",
            "antenna_power", "antenna_energy_injected", "max_abs_antenna_current",
            "max_speed", "max_abs_divB", "l1_divB", "max_abs_By", "min_rho", "min_p",
            "plasma_mass", "plasma_energy", "plasma_magnetic_energy", "plasma_kinetic_energy",
            "plasma_thermal_energy", "solid_magnetic_energy", "solid_constraint_energy",
            "boundary_energy_outflow", "boundary_mass_outflow", "solid_force_x", "solid_force_y",
            "solid_force_z", "solid_impulse_x", "solid_impulse_y", "solid_impulse_z", "max_wall_mass_flux",
            "absorption_energy_exchange", "absorption_mass_exchange",
            "absorption_momentum_exchange_x", "absorption_momentum_exchange_y", "absorption_momentum_exchange_z"
        };
        return names;
    }

    void write_fixed_1d(const std::string& name, const std::vector<double>& values) {
        const hsize_t dimension = values.size();
        const hid_t space = h5_id(H5Screate_simple(1, &dimension, nullptr), "creating coordinate space");
        const hid_t dataset = h5_id(
            H5Dcreate2(file_, name.c_str(), H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
            "creating coordinate " + name);
        const herr_t status = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
        H5Dclose(dataset); H5Sclose(space);
        h5_check(status, "writing coordinate " + name);
    }

    hid_t create_series(const std::string& name, hid_t location = -1) {
        if (location < 0) location = file_;
        const hsize_t initial[1] = {0}, maximum[1] = {H5S_UNLIMITED}, chunk[1] = {128};
        const hid_t space = h5_id(H5Screate_simple(1, initial, maximum), "creating series space");
        const hid_t properties = h5_id(H5Pcreate(H5P_DATASET_CREATE), "creating series properties");
        h5_check(H5Pset_chunk(properties, 1, chunk), "chunking series");
        const hid_t dataset = h5_id(
            H5Dcreate2(location, name.c_str(), H5T_IEEE_F64LE, space, H5P_DEFAULT, properties, H5P_DEFAULT),
            "creating series " + name);
        H5Pclose(properties); H5Sclose(space);
        return dataset;
    }

    hid_t create_field(hid_t group, const std::string& name) {
        const hsize_t initial[3] = {0, static_cast<hsize_t>(ny_), static_cast<hsize_t>(nx_)};
        const hsize_t maximum[3] = {H5S_UNLIMITED, static_cast<hsize_t>(ny_), static_cast<hsize_t>(nx_)};
        const hsize_t chunk[3] = {1, std::min<hsize_t>(64, ny_), std::min<hsize_t>(64, nx_)};
        const hid_t space = h5_id(H5Screate_simple(3, initial, maximum), "creating field space");
        const hid_t properties = h5_id(H5Pcreate(H5P_DATASET_CREATE), "creating field properties");
        h5_check(H5Pset_chunk(properties, 3, chunk), "chunking field " + name);
        if (H5Zfilter_avail(H5Z_FILTER_DEFLATE) > 0) {
            h5_check(H5Pset_shuffle(properties), "enabling shuffle filter");
            h5_check(H5Pset_deflate(properties, 4), "enabling compression");
        }
        const hid_t dataset = h5_id(
            H5Dcreate2(group, name.c_str(), H5T_IEEE_F64LE, space, H5P_DEFAULT, properties, H5P_DEFAULT),
            "creating field " + name);
        H5Pclose(properties); H5Sclose(space);
        return dataset;
    }

    void append_scalar(hid_t dataset, double value) {
        const hsize_t extent[1] = {frame_ + 1};
        h5_check(H5Dset_extent(dataset, extent), "extending scalar series");
        const hid_t file_space = h5_id(H5Dget_space(dataset), "opening scalar series space");
        const hsize_t start[1] = {frame_}, count[1] = {1};
        h5_check(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr, count, nullptr),
                 "selecting scalar series position");
        const hid_t memory_space = h5_id(H5Screate_simple(1, count, nullptr), "creating scalar memory space");
        const herr_t status = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, memory_space, file_space, H5P_DEFAULT, &value);
        H5Sclose(memory_space); H5Sclose(file_space);
        h5_check(status, "appending scalar series");
    }

    void append_field(hid_t dataset, const std::vector<double>& values) {
        if (values.size() != static_cast<std::size_t>(nx_ * ny_)) {
            throw std::runtime_error("Output field has the wrong number of cells");
        }
        const hsize_t extent[3] = {frame_ + 1, static_cast<hsize_t>(ny_), static_cast<hsize_t>(nx_)};
        h5_check(H5Dset_extent(dataset, extent), "extending field");
        const hid_t file_space = h5_id(H5Dget_space(dataset), "opening field space");
        const hsize_t start[3] = {frame_, 0, 0};
        const hsize_t count[3] = {1, static_cast<hsize_t>(ny_), static_cast<hsize_t>(nx_)};
        h5_check(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr, count, nullptr),
                 "selecting field frame");
        const hid_t memory_space = h5_id(H5Screate_simple(3, count, nullptr), "creating field memory space");
        const herr_t status = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, memory_space, file_space,
                                       H5P_DEFAULT, values.data());
        H5Sclose(memory_space); H5Sclose(file_space);
        h5_check(status, "appending field");
    }
};

struct Primitive {
    double density, u, v, w, pressure, bx_value, by_value, bz_value, psi_value;
};

class Solver {
public:
    Solver(Config config, InitialData initial, AntennaData antenna, std::vector<double> solid_mask)
        : c_(std::move(config)), dx_(c_.lx / c_.nx), dy_(c_.ly / c_.ny),
          periodic_x_(c_.boundary_x == "periodic"), periodic_y_(c_.boundary_y == "periodic"),
          absorbing_(c_.boundary_x == "absorbing" || c_.boundary_y == "absorbing"),
          cells_(static_cast<std::size_t>(c_.nx * c_.ny)), antenna_(std::move(antenna)),
          solid_mask_(std::move(solid_mask)),
          executor_(c_.requested_threads) {
        initialize(initial);
        if (!solid_mask_.empty()) {
            solid_density_ = initial.rho;
            solid_thermal_energy_.resize(cells_.size());
            for (std::size_t cell = 0; cell < cells_.size(); ++cell) {
                solid_thermal_energy_[cell] = initial.p[cell] / (c_.gamma - 1.0);
            }
        }
        if (absorbing_) {
            absorption_profile_.resize(cells_.size(), 0.0);
            for (int j = 0; j < c_.ny; ++j) {
                for (int i = 0; i < c_.nx; ++i) {
                    double ramp = 0.0;
                    if (c_.boundary_x == "absorbing") {
                        const double distance = std::min((i + 0.5) * dx_, c_.lx - (i + 0.5) * dx_);
                        ramp = std::max(ramp, 1.0 - distance / c_.absorption_width);
                    }
                    if (c_.boundary_y == "absorbing") {
                        const double distance = std::min((j + 0.5) * dy_, c_.ly - (j + 0.5) * dy_);
                        ramp = std::max(ramp, 1.0 - distance / c_.absorption_width);
                    }
                    const std::size_t cell = index(i, j);
                    absorption_profile_[cell] = ramp * ramp;
                    max_absorption_profile_ = std::max(max_absorption_profile_, absorption_profile_[cell]);
                    if (ramp > 0.0 && is_solid(cell)) {
                        throw std::runtime_error("The active absorption layer must not overlap a stationary solid");
                    }
                }
            }
            if (!(max_absorption_profile_ > 0.0)) {
                throw std::runtime_error("The absorption width must include at least one active cell center");
            }
        }
        if (c_.antenna_model == "prescribed_current") initialize_prescribed_current(initial);
        else if (c_.antenna_model != "resistive_impressed")
            throw std::runtime_error("antenna_model must be resistive_impressed or prescribed_current");
    }

    void run(const std::string& output_filename) {
        Hdf5Output output(output_filename, c_, antenna_, executor_.thread_count(), solid_mask_);
        std::cout << "Running on " << c_.nx << 'x' << c_.ny << " cells with "
                  << executor_.thread_count() << " threads, eta=" << c_.resistivity
                  << ", antenna=" << (antenna_.enabled() ? "on" : "off") << '\n';

        double time = 0.0;
        int step = 0;
        double next_output = c_.output_interval;
        write_frame(output, step, time, 0.0);
        using WallClock = std::chrono::steady_clock;
        const auto wall_start = WallClock::now();
        auto previous_save_wall = wall_start;

        while (time < c_.t_end && step < c_.max_steps) {
            const double ch = cleaning_speed(cells_);
            double dt = stable_timestep(cells_, ch);
            dt = std::min(dt, c_.t_end - time);
            if (next_output > time) dt = std::min(dt, next_output - time);
            if (current_operator_) {
                const double tol = 64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(time));
                const auto knot = std::upper_bound(antenna_.time.begin(), antenna_.time.end(), time + tol);
                if (knot != antenna_.time.end()) dt = std::min(dt, *knot - time);
            }
            if (!(dt > 0.0) || !std::isfinite(dt)) throw std::runtime_error("Invalid time step");

            if (current_operator_) advance_prescribed_current(dt, ch, time);
            else advance_ssprk2(dt, ch, time);
            time += dt;
            ++step;
            const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
                                     std::max(1.0, std::abs(time));
            if (c_.t_end - time >= 0.0 && c_.t_end - time <= tolerance) time = c_.t_end;
            if (time + tolerance >= next_output || time + tolerance >= c_.t_end) {
                write_frame(output, step, time, dt);
                const auto current_wall = WallClock::now();
                const double interval_wall_seconds =
                    std::chrono::duration<double>(current_wall - previous_save_wall).count();
                const double elapsed_wall_seconds =
                    std::chrono::duration<double>(current_wall - wall_start).count();
                const double remaining_simulation_time = std::max(0.0, c_.t_end - time);
                const double eta_seconds = time > 0.0
                    ? elapsed_wall_seconds * remaining_simulation_time / time
                    : std::numeric_limits<double>::infinity();
                std::cout << "step=" << step << " time=" << std::setprecision(8) << time
                          << " dt=" << dt
                          << " interval_wall=" << format_wall_time(interval_wall_seconds)
                          << " elapsed=" << format_wall_time(elapsed_wall_seconds)
                          << " ETA=" << format_wall_time(eta_seconds) << '\n' << std::flush;
                previous_save_wall = current_wall;
                while (next_output <= time + tolerance) next_output += c_.output_interval;
            }
        }
        if (step >= c_.max_steps && time < c_.t_end) {
            throw std::runtime_error("max_steps reached before t_end");
        }
        const double total_wall_seconds =
            std::chrono::duration<double>(WallClock::now() - wall_start).count();
        std::cout << "Completed " << step << " steps at t=" << std::setprecision(12) << time
                  << " in " << format_wall_time(total_wall_seconds)
                  << "; wrote " << output_filename << '\n' << std::flush;
    }

private:
    Config c_;
    double dx_, dy_;
    bool periodic_x_, periodic_y_, absorbing_;
    std::vector<State> cells_;
    AntennaData antenna_;
    std::vector<double> solid_mask_, solid_density_, solid_thermal_energy_;
    std::vector<double> absorption_profile_;
    double max_absorption_profile_ = 0.0;
    ParallelExecutor executor_;
    double antenna_energy_injected_ = 0.0;
    double solid_constraint_energy_ = 0.0;
    double boundary_energy_outflow_ = 0.0, boundary_mass_outflow_ = 0.0;
    double absorption_energy_exchange_ = 0.0, absorption_mass_exchange_ = 0.0;
    std::array<double, 3> absorption_momentum_exchange_{};
    std::array<double, 3> solid_force_{}, solid_impulse_{};
    double max_wall_mass_flux_ = 0.0;
    std::unique_ptr<PrescribedCurrentOperator> current_operator_;
    std::vector<double> potential_;
    const std::vector<double>* active_potential_ = nullptr;
    double max_stage_current_constraint_error_ = 0.0;
    double last_circuit_power_ = 0.0;
    std::vector<double> current_derivative_, potential_rhs_;

    struct RhsRates {
        double antenna_power = 0.0;
        double boundary_energy = 0.0, boundary_mass = 0.0;
        std::array<double, 3> solid_force{};
        double max_wall_mass_flux = 0.0;
    };

    bool is_solid(std::size_t cell) const {
        return !solid_mask_.empty() && solid_mask_[cell] == 1.0;
    }

    // Freeze only the mechanical/thermal placeholder state.  Magnetic fields
    // and GLM cleaning remain evolved in the finite-conductivity body.
    double constrain_solid(State& q, std::size_t cell) const {
        if (!is_solid(cell)) return 0.0;
        const double previous_energy = q[total_energy];
        q[rho] = solid_density_[cell];
        q[mx] = q[my] = q[mz] = 0.0;
        q[total_energy] = solid_thermal_energy_[cell] +
            0.5 * (q[bx] * q[bx] + q[by] * q[by] + q[bz] * q[bz]);
        return q[total_energy] - previous_energy;
    }

    struct AntennaSlice {
        std::size_t lower = 0;
        std::size_t upper = 0;
        double upper_weight = 0.0;
        bool active = false;
    };

    int wrapped(int value, int extent) const {
        value %= extent;
        return value < 0 ? value + extent : value;
    }

    std::size_t index(int i, int j) const {
        const int column = periodic_x_ ? wrapped(i, c_.nx) :
            std::clamp(i, 0, c_.nx - 1);
        const int row = periodic_y_ ? wrapped(j, c_.ny) :
            std::clamp(j, 0, c_.ny - 1);
        return static_cast<std::size_t>(row * c_.nx + column);
    }

    AntennaSlice antenna_slice(double time) const {
        AntennaSlice slice;
        if (!antenna_.enabled() || time < antenna_.time.front() || time > antenna_.time.back()) {
            return slice;
        }
        slice.active = true;
        if (antenna_.sample_count() == 1 || time <= antenna_.time.front()) return slice;
        if (time >= antenna_.time.back()) {
            slice.lower = slice.upper = antenna_.sample_count() - 1;
            return slice;
        }
        const auto upper = std::upper_bound(antenna_.time.begin(), antenna_.time.end(), time);
        slice.upper = static_cast<std::size_t>(upper - antenna_.time.begin());
        slice.lower = slice.upper - 1;
        slice.upper_weight = (time - antenna_.time[slice.lower]) /
            (antenna_.time[slice.upper] - antenna_.time[slice.lower]);
        return slice;
    }

    std::array<double, 3> antenna_current(const AntennaSlice& slice,
                                          std::size_t cell) const {
        if (!slice.active) return {0.0, 0.0, 0.0};
        const std::size_t cells_per_sample = cells_.size();
        const std::size_t lower_offset = slice.lower * cells_per_sample + cell;
        const std::size_t upper_offset = slice.upper * cells_per_sample + cell;
        const double lower_weight = 1.0 - slice.upper_weight;
        auto interpolate = [&](const std::vector<double>& values) {
            return lower_weight * values[lower_offset] +
                   slice.upper_weight * values[upper_offset];
        };
        return {interpolate(antenna_.jx), interpolate(antenna_.jy),
                interpolate(antenna_.jz)};
    }

    Primitive primitive(const State& q) const {
        if (!(q[rho] > 0.0)) throw std::runtime_error("Non-positive density");
        Primitive p{};
        p.density = q[rho];
        p.u = q[mx] / q[rho]; p.v = q[my] / q[rho]; p.w = q[mz] / q[rho];
        p.bx_value = q[bx]; p.by_value = q[by]; p.bz_value = q[bz];
        p.psi_value = q[psi];
        const double kinetic = 0.5 * p.density * (p.u * p.u + p.v * p.v + p.w * p.w);
        const double magnetic = 0.5 * (p.bx_value * p.bx_value + p.by_value * p.by_value +
                                       p.bz_value * p.bz_value);
        p.pressure = (c_.gamma - 1.0) * (q[total_energy] - kinetic - magnetic);
        return p;
    }

    State ambient_state() const {
        State q{};
        q[rho] = c_.ambient[0];
        for (int component = 0; component < 3; ++component) {
            q[mx + component] = q[rho] * c_.ambient[2 + component];
            q[bx + component] = c_.ambient[5 + component];
        }
        q[total_energy] = c_.ambient[1] / (c_.gamma - 1.0) +
            0.5 * q[rho] * (c_.ambient[2] * c_.ambient[2] +
                            c_.ambient[3] * c_.ambient[3] + c_.ambient[4] * c_.ambient[4]) +
            0.5 * (q[bx] * q[bx] + q[by] * q[by] + q[bz] * q[bz]);
        return q;
    }

    void initialize(const InitialData& initial) {
        executor_.for_each(cells_.size(), [&](std::size_t cell) {
            const double density = initial.rho[cell];
            const double pressure = initial.p[cell];
            if (!(density >= c_.rho_floor) || !(pressure >= c_.p_floor) ||
                !std::isfinite(density) || !std::isfinite(pressure)) {
                throw std::runtime_error("Initial density or pressure is non-physical");
            }
            State q{};
            q[rho] = density;
            q[mx] = density * initial.u[cell];
            q[my] = density * initial.v[cell];
            q[mz] = density * initial.w[cell];
            q[bx] = initial.bx[cell]; q[by] = initial.by[cell]; q[bz] = initial.bz[cell];
            q[total_energy] = pressure / (c_.gamma - 1.0) +
                              0.5 * density * (initial.u[cell] * initial.u[cell] +
                                               initial.v[cell] * initial.v[cell] +
                                               initial.w[cell] * initial.w[cell]) +
                              0.5 * (initial.bx[cell] * initial.bx[cell] +
                                     initial.by[cell] * initial.by[cell] +
                                     initial.bz[cell] * initial.bz[cell]);
            q[psi] = 0.0;
            if (!std::all_of(q.begin(), q.end(), [](double value) { return std::isfinite(value); })) {
                throw std::runtime_error("Initial profile contains a non-finite value");
            }
            cells_[cell] = q;
        });
    }

    double minmod(double a, double b, double c) const {
        if (a > 0.0 && b > 0.0 && c > 0.0) return std::min({a, b, c});
        if (a < 0.0 && b < 0.0 && c < 0.0) return std::max({a, b, c});
        return 0.0;
    }

    State slope(const std::vector<State>& data, int i, int j, int direction) const {
        State result{};
        if (!c_.second_order) return result;
        if (!solid_mask_.empty() &&
            (is_solid(index(i, j)) ||
             is_solid(index(i - (direction == 0), j - (direction == 1))) ||
             is_solid(index(i + (direction == 0), j + (direction == 1))))) return result;
        const State center = reconstruction_state(data[index(i, j)]);
        const State lower = reconstruction_state(data[index(i - (direction == 0), j - (direction == 1))]);
        const State upper = reconstruction_state(data[index(i + (direction == 0), j + (direction == 1))]);
        for (std::size_t k = 0; k < nvar; ++k) {
            const double backward = center[k] - lower[k];
            const double forward = upper[k] - center[k];
            result[k] = minmod(c_.limiter_theta * backward, 0.5 * (backward + forward),
                               c_.limiter_theta * forward);
        }
        return result;
    }

    State reconstruction_state(const State& q) const {
        if (!current_operator_) return q;
        const Primitive p = primitive(q);
        State r = q;
        r[mx] = p.u; r[my] = p.v; r[mz] = p.w;
        r[total_energy] = p.pressure;
        return r;
    }

    State reconstructed_state(State r, int direction, double normal_b) const {
        if (!current_operator_) return r;
        Primitive p{r[rho], r[mx], r[my], r[mz], r[total_energy], r[bx], r[by], r[bz], 0.0};
        return state_from_primitive(p, direction, normal_b);
    }

    double ct_normal_field(int i, int j, int direction) const {
        const auto& a = *active_potential_;
        const auto ni = [&](int x, int y) { return current_operator_->node_index(x, y); };
        return direction == 0 ? (a[ni(i + 1, j + 1)] - a[ni(i + 1, j)]) / dy_ :
                               -(a[ni(i + 1, j + 1)] - a[ni(i, j + 1)]) / dx_;
    }

    bool admissible(const State& q) const {
        if (!(q[rho] >= c_.rho_floor)) return false;
        const Primitive p = primitive(q);
        return p.pressure >= c_.p_floor &&
               std::all_of(q.begin(), q.end(), [](double value) { return std::isfinite(value); });
    }

    double fast_speed(const Primitive& p, int direction) const {
        const double sound2 = c_.gamma * p.pressure / p.density;
        const double magnetic2 = (p.bx_value * p.bx_value + p.by_value * p.by_value +
                                  p.bz_value * p.bz_value) / p.density;
        const double normal_b = direction == 0 ? p.bx_value : p.by_value;
        const double normal2 = normal_b * normal_b / p.density;
        const double sum = sound2 + magnetic2;
        const double discriminant = std::max(0.0, sum * sum - 4.0 * sound2 * normal2);
        return std::sqrt(0.5 * (sum + std::sqrt(discriminant)));
    }

    State physical_flux(const State& q, int direction, double ch) const {
        const Primitive p = primitive(q);
        const double velocity[3] = {p.u, p.v, p.w};
        const double magnetic[3] = {p.bx_value, p.by_value, p.bz_value};
        const double vn = velocity[direction], bn = magnetic[direction];
        const double total_pressure = p.pressure + 0.5 *
            (p.bx_value * p.bx_value + p.by_value * p.by_value + p.bz_value * p.bz_value);
        State flux{};
        flux[rho] = p.density * vn;
        for (int component = 0; component < 3; ++component) {
            flux[mx + component] = p.density * vn * velocity[component] - bn * magnetic[component];
            if (component == direction) flux[mx + component] += total_pressure;
            flux[bx + component] = vn * magnetic[component] - bn * velocity[component];
        }
        flux[bx + direction] = p.psi_value;
        const double velocity_dot_magnetic = p.u * p.bx_value + p.v * p.by_value + p.w * p.bz_value;
        flux[total_energy] = (q[total_energy] + total_pressure) * vn - bn * velocity_dot_magnetic;
        flux[psi] = ch * ch * bn;
        return flux;
    }

    State rusanov_flux(const State& left, const State& right, int direction, double ch) const {
        const Primitive pl = primitive(left), pr = primitive(right);
        const double vnl = direction == 0 ? pl.u : pl.v;
        const double vnr = direction == 0 ? pr.u : pr.v;
        const double wave_speed = std::max({std::abs(vnl) + fast_speed(pl, direction),
                                            std::abs(vnr) + fast_speed(pr, direction), ch});
        const State fl = physical_flux(left, direction, ch);
        const State fr = physical_flux(right, direction, ch);
        State flux{};
        for (std::size_t k = 0; k < nvar; ++k) {
            flux[k] = 0.5 * (fl[k] + fr[k]) - 0.5 * wave_speed * (right[k] - left[k]);
        }
        return flux;
    }

    State state_from_primitive(const Primitive& p, int direction, double normal_b) const {
        const double velocity[3] = {p.u, p.v, p.w};
        double magnetic[3] = {p.bx_value, p.by_value, p.bz_value};
        magnetic[direction] = normal_b;
        State q{};
        q[rho] = p.density;
        for (int component = 0; component < 3; ++component) {
            q[mx + component] = p.density * velocity[component];
            q[bx + component] = magnetic[component];
        }
        q[total_energy] = p.pressure / (c_.gamma - 1.0) +
                          0.5 * p.density * (p.u * p.u + p.v * p.v + p.w * p.w) +
                          0.5 * (magnetic[0] * magnetic[0] + magnetic[1] * magnetic[1] +
                                 magnetic[2] * magnetic[2]);
        q[psi] = 0.0;
        return q;
    }

    State hlld_flux(const State& left, const State& right, int direction, double ch) const {
        const Primitive original_left = primitive(left);
        const Primitive original_right = primitive(right);
        const double bn_left = direction == 0 ? original_left.bx_value : original_left.by_value;
        const double bn_right = direction == 0 ? original_right.bx_value : original_right.by_value;
        const double bn_star = 0.5 * (bn_left + bn_right) -
                               0.5 * (original_right.psi_value - original_left.psi_value) / ch;
        const double psi_star = 0.5 * (original_left.psi_value + original_right.psi_value) -
                                0.5 * ch * (bn_right - bn_left);

        const State ul = state_from_primitive(original_left, direction, bn_star);
        const State ur = state_from_primitive(original_right, direction, bn_star);
        const Primitive pl = primitive(ul), pr = primitive(ur);
        const State fl = physical_flux(ul, direction, ch);
        const State fr = physical_flux(ur, direction, ch);
        const int normal = direction;
        const int tangent1 = 1 - direction;
        const int tangent2 = 2;
        const double velocity_left[3] = {pl.u, pl.v, pl.w};
        const double velocity_right[3] = {pr.u, pr.v, pr.w};
        const double magnetic_left[3] = {pl.bx_value, pl.by_value, pl.bz_value};
        const double magnetic_right[3] = {pr.bx_value, pr.by_value, pr.bz_value};
        const double vn_left = velocity_left[normal], vn_right = velocity_right[normal];
        const double pt_left = pl.pressure + 0.5 *
            (pl.bx_value * pl.bx_value + pl.by_value * pl.by_value + pl.bz_value * pl.bz_value);
        const double pt_right = pr.pressure + 0.5 *
            (pr.bx_value * pr.bx_value + pr.by_value * pr.by_value + pr.bz_value * pr.bz_value);
        const double sl = std::min(vn_left - fast_speed(pl, direction),
                                   vn_right - fast_speed(pr, direction));
        const double sr = std::max(vn_left + fast_speed(pl, direction),
                                   vn_right + fast_speed(pr, direction));

        auto finish_glm = [&](State flux) {
            flux[bx + normal] = psi_star;
            flux[psi] = ch * ch * bn_star;
            return flux;
        };
        if (sl >= 0.0) return finish_glm(fl);
        if (sr <= 0.0) return finish_glm(fr);

        auto fallback = [&] { return finish_glm(rusanov_flux(ul, ur, direction, ch)); };
        const double contact_denominator = pl.density * (sl - vn_left) -
                                           pr.density * (sr - vn_right);
        if (std::abs(contact_denominator) < 1.0e-14 || !std::isfinite(contact_denominator)) {
            return fallback();
        }
        const double sm = (pl.density * vn_left * (sl - vn_left) -
                           pr.density * vn_right * (sr - vn_right) - pt_left + pt_right) /
                          contact_denominator;
        const double pt_star_left = pt_left + pl.density * (sl - vn_left) * (sm - vn_left);
        const double pt_star_right = pt_right + pr.density * (sr - vn_right) * (sm - vn_right);
        const double pt_star = 0.5 * (pt_star_left + pt_star_right);
        if (!std::isfinite(sm) || !std::isfinite(pt_star)) return fallback();

        auto make_star = [&](const State& u, const Primitive& p, const double velocity[3],
                             const double magnetic[3], double wave_speed) {
            const double vn = velocity[normal];
            if (std::abs(wave_speed - sm) < 1.0e-14) {
                return std::pair<State, bool>{State{}, false};
            }
            const double density_star = p.density * (wave_speed - vn) / (wave_speed - sm);
            const double denominator = p.density * (wave_speed - vn) * (wave_speed - sm) -
                                       bn_star * bn_star;
            if (!(density_star > 0.0) || std::abs(denominator) < 1.0e-14) {
                return std::pair<State, bool>{State{}, false};
            }
            double velocity_star[3] = {velocity[0], velocity[1], velocity[2]};
            double magnetic_star[3] = {magnetic[0], magnetic[1], magnetic[2]};
            velocity_star[normal] = sm;
            magnetic_star[normal] = bn_star;
            for (const int tangent : {tangent1, tangent2}) {
                velocity_star[tangent] = velocity[tangent] -
                    bn_star * magnetic[tangent] * (sm - vn) / denominator;
                magnetic_star[tangent] = magnetic[tangent] *
                    (p.density * (wave_speed - vn) * (wave_speed - vn) - bn_star * bn_star) /
                    denominator;
            }
            const double velocity_dot_magnetic = velocity[0] * magnetic[0] +
                velocity[1] * magnetic[1] + velocity[2] * magnetic[2];
            const double star_dot = velocity_star[0] * magnetic_star[0] +
                velocity_star[1] * magnetic_star[1] + velocity_star[2] * magnetic_star[2];
            const double total_pressure = p.pressure + 0.5 *
                (magnetic[0] * magnetic[0] + magnetic[1] * magnetic[1] + magnetic[2] * magnetic[2]);
            const double energy_star = ((wave_speed - vn) * u[total_energy] - total_pressure * vn +
                pt_star * sm + bn_star * (velocity_dot_magnetic - star_dot)) /
                (wave_speed - sm);
            State star{};
            star[rho] = density_star;
            for (int component = 0; component < 3; ++component) {
                star[mx + component] = density_star * velocity_star[component];
                star[bx + component] = magnetic_star[component];
            }
            star[total_energy] = energy_star;
            star[psi] = 0.0;
            return std::pair<State, bool>{star, admissible(star)};
        };

        const auto [ul_star, left_ok] = make_star(ul, pl, velocity_left, magnetic_left, sl);
        const auto [ur_star, right_ok] = make_star(ur, pr, velocity_right, magnetic_right, sr);
        if (!left_ok || !right_ok) return fallback();
        State fl_star{}, fr_star{};
        for (std::size_t variable = 0; variable < nvar; ++variable) {
            fl_star[variable] = fl[variable] + sl * (ul_star[variable] - ul[variable]);
            fr_star[variable] = fr[variable] + sr * (ur_star[variable] - ur[variable]);
        }

        // For B_n -> 0 the rotational waves merge with the contact wave; the
        // two star states give the regular HLLC limit and exactly preserve a
        // stationary, pressure-balanced tangential discontinuity.
        if (std::abs(bn_star) < 1.0e-12) {
            return finish_glm(sm >= 0.0 ? fl_star : fr_star);
        }

        const double density_left_star = ul_star[rho];
        const double density_right_star = ur_star[rho];
        const double sqrt_left = std::sqrt(density_left_star);
        const double sqrt_right = std::sqrt(density_right_star);
        const double sign_bn = std::copysign(1.0, bn_star);
        const double sl_star = sm - std::abs(bn_star) / sqrt_left;
        const double sr_star = sm + std::abs(bn_star) / sqrt_right;
        const double sum_sqrt = sqrt_left + sqrt_right;

        double velocity_left_star[3], velocity_right_star[3];
        double magnetic_left_star[3], magnetic_right_star[3];
        for (int component = 0; component < 3; ++component) {
            velocity_left_star[component] = ul_star[mx + component] / density_left_star;
            velocity_right_star[component] = ur_star[mx + component] / density_right_star;
            magnetic_left_star[component] = ul_star[bx + component];
            magnetic_right_star[component] = ur_star[bx + component];
        }
        double velocity_double[3] = {0.0, 0.0, 0.0};
        double magnetic_double[3] = {0.0, 0.0, 0.0};
        velocity_double[normal] = sm;
        magnetic_double[normal] = bn_star;
        for (const int tangent : {tangent1, tangent2}) {
            velocity_double[tangent] = (sqrt_left * velocity_left_star[tangent] +
                sqrt_right * velocity_right_star[tangent] +
                sign_bn * (magnetic_right_star[tangent] - magnetic_left_star[tangent])) / sum_sqrt;
            magnetic_double[tangent] = (sqrt_left * magnetic_right_star[tangent] +
                sqrt_right * magnetic_left_star[tangent] + sign_bn * sqrt_left * sqrt_right *
                (velocity_right_star[tangent] - velocity_left_star[tangent])) / sum_sqrt;
        }
        const double dot_double = velocity_double[0] * magnetic_double[0] +
            velocity_double[1] * magnetic_double[1] + velocity_double[2] * magnetic_double[2];
        const double dot_left_star = velocity_left_star[0] * magnetic_left_star[0] +
            velocity_left_star[1] * magnetic_left_star[1] + velocity_left_star[2] * magnetic_left_star[2];
        const double dot_right_star = velocity_right_star[0] * magnetic_right_star[0] +
            velocity_right_star[1] * magnetic_right_star[1] + velocity_right_star[2] * magnetic_right_star[2];

        State ul_double{}, ur_double{};
        ul_double[rho] = density_left_star;
        ur_double[rho] = density_right_star;
        for (int component = 0; component < 3; ++component) {
            ul_double[mx + component] = density_left_star * velocity_double[component];
            ur_double[mx + component] = density_right_star * velocity_double[component];
            ul_double[bx + component] = magnetic_double[component];
            ur_double[bx + component] = magnetic_double[component];
        }
        ul_double[total_energy] = ul_star[total_energy] -
            sqrt_left * sign_bn * (dot_left_star - dot_double);
        ur_double[total_energy] = ur_star[total_energy] +
            sqrt_right * sign_bn * (dot_right_star - dot_double);
        ul_double[psi] = 0.0;
        ur_double[psi] = 0.0;
        if (!admissible(ul_double) || !admissible(ur_double)) return fallback();

        State fl_double{}, fr_double{};
        for (std::size_t variable = 0; variable < nvar; ++variable) {
            fl_double[variable] = fl_star[variable] +
                                  sl_star * (ul_double[variable] - ul_star[variable]);
            fr_double[variable] = fr_star[variable] +
                                  sr_star * (ur_double[variable] - ur_star[variable]);
        }
        if (sl_star >= 0.0) return finish_glm(fl_star);
        if (sm >= 0.0) return finish_glm(fl_double);
        if (sr_star > 0.0) return finish_glm(fr_double);
        return finish_glm(fr_star);
    }

    State material_face_flux(const State& left, const State& right, bool left_solid,
                             bool right_solid, int direction, double ch) const {
        State flux{};
        const Primitive pl = primitive(left), pr = primitive(right);
        const double field_left[3] = {pl.bx_value, pl.by_value, pl.bz_value};
        const double field_right[3] = {pr.bx_value, pr.by_value, pr.bz_value};
        double field_face[3]{};
        for (int component = 0; component < 3; ++component) {
            field_face[component] = 0.5 * (field_left[component] + field_right[component]);
        }

        if (left_solid != right_solid) {
            // A reflected fluid ghost determines the gas pressure at a fixed
            // slip wall.  No fluid mass or advected energy crosses this face.
            const State& fluid = left_solid ? right : left;
            const Primitive pf = left_solid ? pr : pl;
            State mirror = fluid;
            mirror[mx + direction] = -mirror[mx + direction];
            const State mechanical = left_solid ? hlld_flux(mirror, fluid, direction, ch) :
                                                         hlld_flux(fluid, mirror, direction, ch);
            const double fluid_field[3] = {pf.bx_value, pf.by_value, pf.bz_value};
            double fluid_magnetic = 0.0, face_magnetic = 0.0;
            for (int component = 0; component < 3; ++component) {
                fluid_magnetic += 0.5 * fluid_field[component] * fluid_field[component];
                face_magnetic += 0.5 * field_face[component] * field_face[component];
            }
            const double gas_pressure = std::max(c_.p_floor, mechanical[mx + direction] -
                fluid_magnetic + fluid_field[direction] * fluid_field[direction]);
            for (int component = 0; component < 3; ++component) {
                flux[mx + component] = -field_face[direction] * field_face[component];
                if (component == direction) flux[mx + component] += gas_pressure + face_magnetic;
            }
        }

        // Electromagnetic coupling uses the actual body and fluid fields, not
        // the reflected fluid ghost.  Body velocity is zero; fluid has zero
        // normal wall velocity and retains its tangential slip velocity.
        // The LF term stabilizes the plasma/body interface; body/body faces
        // have only magnetic diffusion and the GLM subsystem.
        const double velocity_left[3] = {pl.u, pl.v, pl.w};
        const double velocity_right[3] = {pr.u, pr.v, pr.w};
        const double interface_speed = left_solid != right_solid ? ch : 0.0;
        const double inverse_spacing = 1.0 / (direction == 0 ? dx_ : dy_);
        for (int component = 0; component < 3; ++component) {
            if (component != direction) {
                const double velocity = 0.5 * ((left_solid ? 0.0 : velocity_left[component]) +
                                               (right_solid ? 0.0 : velocity_right[component]));
                flux[bx + component] = -field_face[direction] * velocity -
                    0.5 * interface_speed * (field_right[component] - field_left[component]);
            }
            if (!current_operator_) flux[bx + component] -= c_.resistivity * inverse_spacing *
                (field_right[component] - field_left[component]);
        }
        flux[bx + direction] += 0.5 * (pl.psi_value + pr.psi_value) -
            0.5 * ch * (field_right[direction] - field_left[direction]);
        flux[psi] = ch * ch * field_face[direction] - 0.5 * ch * (pr.psi_value - pl.psi_value);
        // Only electromagnetic Poynting energy can enter the rigid body.
        for (int component = 0; component < 3; ++component) {
            if (component != direction) flux[total_energy] +=
                field_face[component] * flux[bx + component];
        }
        return flux;
    }

    State face_flux(const std::vector<State>& data, const std::vector<State>& slopes,
                    int i, int j, int direction, double ch) const {
        const std::size_t left_index = index(i, j);
        const std::size_t right_index = index(i + (direction == 0), j + (direction == 1));
        if (is_solid(left_index) || is_solid(right_index)) {
            State left = data[left_index], right = data[right_index];
            if (current_operator_) {
                const double bn = ct_normal_field(i, j, direction);
                left = state_from_primitive(primitive(left), direction, bn);
                right = state_from_primitive(primitive(right), direction, bn);
            }
            return material_face_flux(left, right,
                                      is_solid(left_index), is_solid(right_index), direction, ch);
        }
        const bool lower_face = direction == 0 ? i < 0 : j < 0;
        const bool upper_face = direction == 0 ? i == c_.nx - 1 : j == c_.ny - 1;
        const bool absorbing_axis = direction == 0 ? c_.boundary_x == "absorbing" :
                                                    c_.boundary_y == "absorbing";
        if (absorbing_axis && (lower_face || upper_face)) {
            const State& interior = data[lower_face ? right_index : left_index];
            const Primitive pi = primitive(interior);
            const double normal_velocity = direction == 0 ? pi.u : pi.v;
            const double outward_velocity = (lower_face ? -1.0 : 1.0) * normal_velocity;
            State exterior = outward_velocity >= fast_speed(pi, direction) ? interior : ambient_state();
            const Primitive pe = primitive(exterior);
            // The GLM normal-field pair is extrapolated together.  Resetting
            // psi alone would launch a cleaning-wave reflection.  Rebuild E
            // after copying Bn so the exterior thermal pressure is unchanged.
            exterior[bx + direction] = interior[bx + direction];
            exterior[psi] = interior[psi];
            exterior[total_energy] = pe.pressure / (c_.gamma - 1.0) +
                0.5 * pe.density * (pe.u * pe.u + pe.v * pe.v + pe.w * pe.w) +
                0.5 * (exterior[bx] * exterior[bx] + exterior[by] * exterior[by] + exterior[bz] * exterior[bz]);
            // Physical resistivity retains zero normal gradient at this face.
            if (current_operator_) {
                const double bn = ct_normal_field(i, j, direction);
                const State qi = state_from_primitive(pi, direction, bn);
                exterior = state_from_primitive(primitive(exterior), direction, bn);
                return lower_face ? hlld_flux(exterior, qi, direction, ch) : hlld_flux(qi, exterior, direction, ch);
            }
            return lower_face ? hlld_flux(exterior, interior, direction, ch) :
                                hlld_flux(interior, exterior, direction, ch);
        }
        State left = reconstruction_state(data[left_index]), right = reconstruction_state(data[right_index]);
        for (std::size_t k = 0; k < nvar; ++k) {
            left[k] += 0.5 * slopes[left_index][k];
            right[k] -= 0.5 * slopes[right_index][k];
        }
        if (current_operator_) {
            const double bn = ct_normal_field(i, j, direction);
            left = reconstructed_state(left, direction, bn);
            right = reconstructed_state(right, direction, bn);
        }
        if (!admissible(left)) left = current_operator_ ?
            state_from_primitive(primitive(data[left_index]), direction, ct_normal_field(i, j, direction)) : data[left_index];
        if (!admissible(right)) right = current_operator_ ?
            state_from_primitive(primitive(data[right_index]), direction, ct_normal_field(i, j, direction)) : data[right_index];
        State flux = hlld_flux(left, right, direction, ch);
        if (c_.resistivity > 0.0 && !current_operator_) {
            const double inverse_spacing = 1.0 / (direction == 0 ? dx_ : dy_);
            for (int component = 0; component < 3; ++component) {
                flux[bx + component] -= c_.resistivity * inverse_spacing *
                                        (data[right_index][bx + component] - data[left_index][bx + component]);
            }
        }
        return flux;
    }

    double data_derivative(const std::vector<State>& data, int i, int j,
                           Variable variable, int direction) const {
        if (direction == 0) {
            return (data[index(i + 1, j)][variable] - data[index(i - 1, j)][variable]) /
                   (2.0 * dx_);
        }
        return (data[index(i, j + 1)][variable] - data[index(i, j - 1)][variable]) /
               (2.0 * dy_);
    }

    State antenna_source(const std::vector<State>& data, int i, int j,
                         const AntennaSlice& slice) const {
        State source{};
        if (current_operator_) return source;
        if (!slice.active) return source;
        const std::size_t cell = index(i, j);
        const Primitive p = primitive(data[cell]);
        const std::array<double, 3> ja = antenna_current(slice, cell);
        const double magnetic[3] = {p.bx_value, p.by_value, p.bz_value};

        // curl(B)=J_plasma+J_antenna.  The conservative Maxwell-stress flux
        // already contains curl(B)xB, so subtract the impressed-current force.
        const double force[3] = {
            -(ja[1] * magnetic[2] - ja[2] * magnetic[1]),
            -(ja[2] * magnetic[0] - ja[0] * magnetic[2]),
            -(ja[0] * magnetic[1] - ja[1] * magnetic[0]),
        };
        source[mx] = force[0];
        source[my] = force[1];
        source[mz] = force[2];
        source[total_energy] = p.u * force[0] + p.v * force[1] + p.w * force[2];
        if (is_solid(cell)) {
            // Structural support carries body forces.  It is not a plasma
            // momentum source; electrical antenna work is still accounted.
            source[mx] = source[my] = source[mz] = 0.0;
        }

        if (c_.resistivity > 0.0) {
            const auto current_x_plus = antenna_current(slice, index(i + 1, j));
            const auto current_x_minus = antenna_current(slice, index(i - 1, j));
            const auto current_y_plus = antenna_current(slice, index(i, j + 1));
            const auto current_y_minus = antenna_current(slice, index(i, j - 1));
            const double curl_antenna[3] = {
                (current_y_plus[2] - current_y_minus[2]) / (2.0 * dy_),
                -(current_x_plus[2] - current_x_minus[2]) / (2.0 * dx_),
                (current_x_plus[1] - current_x_minus[1]) / (2.0 * dx_) -
                    (current_y_plus[0] - current_y_minus[0]) / (2.0 * dy_),
            };
            for (int component = 0; component < 3; ++component) {
                source[bx + component] = c_.resistivity * curl_antenna[component];
            }

            const double total_current[3] = {
                data_derivative(data, i, j, bz, 1),
                -data_derivative(data, i, j, bz, 0),
                data_derivative(data, i, j, by, 0) - data_derivative(data, i, j, bx, 1),
            };
            const double plasma_current[3] = {
                total_current[0] - ja[0], total_current[1] - ja[1], total_current[2] - ja[2]
            };
            source[total_energy] -= c_.resistivity *
                (ja[0] * plasma_current[0] + ja[1] * plasma_current[1] +
                 ja[2] * plasma_current[2]);
        }
        return source;
    }

    std::vector<State> right_hand_side(const std::vector<State>& data, double ch,
                                       double time, RhsRates& rates) {
        rates = RhsRates{};
        std::vector<State> sx(data.size()), sy(data.size());
        executor_.for_each(data.size(), [&](std::size_t cell) {
            const int i = static_cast<int>(cell % static_cast<std::size_t>(c_.nx));
            const int j = static_cast<int>(cell / static_cast<std::size_t>(c_.nx));
            sx[cell] = slope(data, i, j, 0);
            sy[cell] = slope(data, i, j, 1);
        });

        std::vector<State> flux_x(data.size()), flux_y(data.size());
        executor_.for_each(data.size(), [&](std::size_t cell) {
            const int i = static_cast<int>(cell % static_cast<std::size_t>(c_.nx));
            const int j = static_cast<int>(cell / static_cast<std::size_t>(c_.nx));
            flux_x[cell] = face_flux(data, sx, i, j, 0, ch);
            flux_y[cell] = face_flux(data, sy, i, j, 1, ch);
        });

        // Interior arrays store each cell's right/top face.  The left/bottom
        // nonperiodic faces are distinct faces, not the first interior face.
        // Outflow uses zero-gradient states; absorbing uses ambient HLLD.
        std::vector<State> left_boundary_flux, lower_boundary_flux;
        if (!periodic_x_) {
            left_boundary_flux.resize(static_cast<std::size_t>(c_.ny));
            executor_.for_each(left_boundary_flux.size(), [&](std::size_t row) {
                left_boundary_flux[row] = face_flux(data, sx, -1, static_cast<int>(row), 0, ch);
            });
        }
        if (!periodic_y_) {
            lower_boundary_flux.resize(static_cast<std::size_t>(c_.nx));
            executor_.for_each(lower_boundary_flux.size(), [&](std::size_t column) {
                lower_boundary_flux[column] = face_flux(data, sy, static_cast<int>(column), -1, 1, ch);
            });
        }

        if (current_operator_) prescribed_induction_fluxes(data, time, flux_x, flux_y,
            left_boundary_flux, lower_boundary_flux, rates);

        if (!periodic_x_) {
            for (int j = 0; j < c_.ny; ++j) {
                const State& outward = flux_x[index(c_.nx - 1, j)];
                const State& inward = left_boundary_flux[static_cast<std::size_t>(j)];
                rates.boundary_energy += (outward[total_energy] - inward[total_energy]) * dy_;
                rates.boundary_mass += (outward[rho] - inward[rho]) * dy_;
            }
        }
        if (!periodic_y_) {
            for (int i = 0; i < c_.nx; ++i) {
                const State& outward = flux_y[index(i, c_.ny - 1)];
                const State& inward = lower_boundary_flux[static_cast<std::size_t>(i)];
                rates.boundary_energy += (outward[total_energy] - inward[total_energy]) * dx_;
                rates.boundary_mass += (outward[rho] - inward[rho]) * dx_;
            }
        }
        if (!solid_mask_.empty()) {
            for (int j = 0; j < c_.ny; ++j) {
                for (int i = 0; i < c_.nx; ++i) {
                    const std::size_t cell = index(i, j);
                    for (int direction = 0; direction < 2; ++direction) {
                        const std::size_t next = index(i + (direction == 0), j + (direction == 1));
                        if (is_solid(cell) == is_solid(next)) continue;
                        const State& flux = direction == 0 ? flux_x[cell] : flux_y[cell];
                        const double area = direction == 0 ? dy_ : dx_;
                        const double sign = is_solid(cell) ? -1.0 : 1.0;
                        for (int component = 0; component < 3; ++component) {
                            rates.solid_force[component] += sign * area * flux[mx + component];
                        }
                        rates.max_wall_mass_flux = std::max(rates.max_wall_mass_flux, std::abs(flux[rho]));
                    }
                }
            }
        }

        const AntennaSlice antenna = antenna_slice(time);
        std::vector<State> rhs(data.size());
        std::vector<double> antenna_power_density(
            antenna.active ? data.size() : 0, 0.0);
        executor_.for_each(data.size(), [&](std::size_t cell) {
            const int i = static_cast<int>(cell % static_cast<std::size_t>(c_.nx));
            const int j = static_cast<int>(cell / static_cast<std::size_t>(c_.nx));
            const State& left_flux = i == 0 && !periodic_x_ ?
                left_boundary_flux[static_cast<std::size_t>(j)] : flux_x[index(i - 1, j)];
            const State& lower_flux = j == 0 && !periodic_y_ ?
                lower_boundary_flux[static_cast<std::size_t>(i)] : flux_y[index(i, j - 1)];
            for (std::size_t k = 0; k < nvar; ++k) {
                rhs[cell][k] = -(flux_x[cell][k] - left_flux[k]) / dx_ -
                                (flux_y[cell][k] - lower_flux[k]) / dy_;
            }
            if (antenna.active) {
                const State source = antenna_source(data, i, j, antenna);
                for (std::size_t k = 0; k < nvar; ++k) rhs[cell][k] += source[k];
                antenna_power_density[cell] = source[total_energy];
            }
        });
        if (!current_operator_) {
            for (const double power_density : antenna_power_density) rates.antenna_power += power_density;
            rates.antenna_power *= dx_ * dy_;
        }
        return rhs;
    }

    double cleaning_speed(const std::vector<State>& data) const {
        double maximum = 0.0;
        for (const State& q : data) {
            const Primitive p = primitive(q);
            maximum = std::max({maximum, std::abs(p.u) + fast_speed(p, 0),
                                std::abs(p.v) + fast_speed(p, 1)});
            if (absorbing_) {
                const Primitive bound = absorption_speed_bound(p);
                maximum = std::max({maximum, bound.u + fast_speed(bound, 0),
                                    bound.v + fast_speed(bound, 1)});
            }
        }
        return c_.glm_speed_factor * std::max(maximum, 1.0e-12);
    }

    Primitive absorption_speed_bound(Primitive p) const {
        // Relaxing density before the MHD step can increase the Alfven speed
        // while B is still present.  Bound the entire fluid relaxation path
        // and the mixed normal-field ambient face state, not only its ends.
        p.density = std::min(p.density, c_.ambient[0]);
        p.pressure = std::max(p.pressure, c_.ambient[1]);
        p.u = std::max(std::abs(p.u), std::abs(c_.ambient[2]));
        p.v = std::max(std::abs(p.v), std::abs(c_.ambient[3]));
        p.w = std::max(std::abs(p.w), std::abs(c_.ambient[4]));
        p.bx_value = std::max(std::abs(p.bx_value), std::abs(c_.ambient[5]));
        p.by_value = std::max(std::abs(p.by_value), std::abs(c_.ambient[6]));
        p.bz_value = std::max(std::abs(p.bz_value), std::abs(c_.ambient[7]));
        return p;
    }

    double stable_timestep(const std::vector<State>& data, double ch) const {
        double speed_x = ch, speed_y = ch;
        for (const State& q : data) {
            const Primitive p = primitive(q);
            speed_x = std::max(speed_x, std::abs(p.u) + fast_speed(p, 0));
            speed_y = std::max(speed_y, std::abs(p.v) + fast_speed(p, 1));
            if (absorbing_) {
                const Primitive bound = absorption_speed_bound(p);
                speed_x = std::max(speed_x, bound.u + fast_speed(bound, 0));
                speed_y = std::max(speed_y, bound.v + fast_speed(bound, 1));
            }
        }
        double dt = c_.cfl / (speed_x / dx_ + speed_y / dy_);
        if (c_.resistivity > 0.0) {
            const double diffusion_dt = c_.diffusion_cfl /
                (c_.resistivity * (1.0 / (dx_ * dx_) + 1.0 / (dy_ * dy_)));
            dt = std::min(dt, diffusion_dt);
        }
        if (absorbing_ && c_.absorption_magnetic_diffusivity > 0.0 && max_absorption_profile_ > 0.0) {
            const double magnetic_dt = c_.diffusion_cfl /
                (c_.absorption_magnetic_diffusivity * max_absorption_profile_ *
                 (1.0 / (dx_ * dx_) + 1.0 / (dy_ * dy_)));
            dt = std::min(dt, magnetic_dt);
        }
        return dt;
    }

    std::vector<State> magnetic_absorption_rhs(const std::vector<State>& data) {
        std::vector<State> electric(data.size());
        executor_.for_each(data.size(), [&](std::size_t cell) {
            const int i = static_cast<int>(cell % static_cast<std::size_t>(c_.nx));
            const int j = static_cast<int>(cell / static_cast<std::size_t>(c_.nx));
            const double eta = c_.absorption_magnetic_diffusivity * absorption_profile_[cell];
            electric[cell][bx] = eta * data_derivative(data, i, j, bz, 1);
            electric[cell][by] = -eta * data_derivative(data, i, j, bz, 0);
            electric[cell][bz] = eta * (data_derivative(data, i, j, by, 0) -
                                       data_derivative(data, i, j, bx, 1));
            // A clamped central derivative has endpoint terms.  Zeroing
            // tangential artificial E at every nonperiodic exterior makes
            // -curl(eta curl B) dissipative in the cell-volume inner product.
            // Project E before curl, so central div(curl E) still vanishes.
            if (!periodic_x_ && (i == 0 || i == c_.nx - 1)) {
                electric[cell][by] = electric[cell][bz] = 0.0;
            }
            if (!periodic_y_ && (j == 0 || j == c_.ny - 1)) {
                electric[cell][bx] = electric[cell][bz] = 0.0;
            }
        });
        std::vector<State> rhs(data.size());
        executor_.for_each(data.size(), [&](std::size_t cell) {
            const int i = static_cast<int>(cell % static_cast<std::size_t>(c_.nx));
            const int j = static_cast<int>(cell / static_cast<std::size_t>(c_.nx));
            rhs[cell][bx] = -data_derivative(electric, i, j, bz, 1);
            rhs[cell][by] = data_derivative(electric, i, j, bz, 0);
            rhs[cell][bz] = -data_derivative(electric, i, j, by, 0) +
                             data_derivative(electric, i, j, bx, 1);
        });
        return rhs;
    }

    void apply_absorption(double duration) {
        if (!absorbing_) return;
        std::vector<double> energy_change(cells_.size(), 0.0), mass_change(cells_.size(), 0.0);
        std::vector<std::array<double, 3>> momentum_change(cells_.size());
        executor_.for_each(cells_.size(), [&](std::size_t cell) {
            if (is_solid(cell) || absorption_profile_[cell] == 0.0) return;
            State& q = cells_[cell];
            const Primitive p = primitive(q);
            const double previous_energy = q[total_energy], previous_mass = q[rho];
            const double relaxed = -std::expm1(-c_.absorption_rate * absorption_profile_[cell] * duration);
            const double retained = 1.0 - relaxed;
            q[rho] = retained * q[rho] + relaxed * c_.ambient[0];
            for (int component = 0; component < 3; ++component) {
                const double previous_momentum = q[mx + component];
                q[mx + component] = retained * q[mx + component] +
                    relaxed * c_.ambient[0] * c_.ambient[2 + component];
                momentum_change[cell][component] = q[mx + component] - previous_momentum;
            }
            const double thermal = (retained * p.pressure + relaxed * c_.ambient[1]) /
                                   (c_.gamma - 1.0);
            q[total_energy] = thermal +
                0.5 * (q[mx] * q[mx] + q[my] * q[my] + q[mz] * q[mz]) / q[rho] +
                0.5 * (q[bx] * q[bx] + q[by] * q[by] + q[bz] * q[bz]);
            energy_change[cell] = q[total_energy] - previous_energy;
            mass_change[cell] = q[rho] - previous_mass;
        });
        if (current_operator_ && c_.absorption_magnetic_diffusivity > 0.0 && max_absorption_profile_ > 0.0)
            apply_prescribed_magnetic_absorption(duration, energy_change);
        if (!current_operator_ && c_.absorption_magnetic_diffusivity > 0.0 && max_absorption_profile_ > 0.0) {
            const std::vector<State> rhs0 = magnetic_absorption_rhs(cells_);
            std::vector<State> stage = cells_;
            executor_.for_each(cells_.size(), [&](std::size_t cell) {
                for (int component = 0; component < 3; ++component) {
                    stage[cell][bx + component] += duration * rhs0[cell][bx + component];
                }
            });
            const std::vector<State> rhs1 = magnetic_absorption_rhs(stage);
            executor_.for_each(cells_.size(), [&](std::size_t cell) {
                State& q = cells_[cell];
                const double previous_energy = q[total_energy];
                double magnetic_change = 0.0;
                for (int component = 0; component < 3; ++component) {
                    const double old_field = q[bx + component];
                    q[bx + component] = 0.5 * (old_field + stage[cell][bx + component] +
                                              duration * rhs1[cell][bx + component]);
                    magnetic_change += 0.5 * (q[bx + component] * q[bx + component] - old_field * old_field);
                }
                // This artificial magnetic sink is not physical resistivity:
                // remove its magnetic energy into the absorber reservoir and
                // preserve local thermal pressure instead of adding heat.
                q[total_energy] += magnetic_change;
                energy_change[cell] += q[total_energy] - previous_energy;
            });
        }
        for (std::size_t cell = 0; cell < cells_.size(); ++cell) {
            absorption_energy_exchange_ += energy_change[cell] * dx_ * dy_;
            absorption_mass_exchange_ += mass_change[cell] * dx_ * dy_;
            for (int component = 0; component < 3; ++component) {
                absorption_momentum_exchange_[component] += momentum_change[cell][component] * dx_ * dy_;
            }
            if (!admissible(cells_[cell])) {
                throw std::runtime_error("State became non-physical in absorption layer");
            }
        }
    }

    std::vector<double> commanded_current(double time) const {
        const auto slice = antenna_slice(time);
        std::vector<double> cell_current(cells_.size());
        for (std::size_t cell = 0; cell < cells_.size(); ++cell)
            cell_current[cell] = antenna_current(slice, cell)[2];
        return current_operator_->project_current(cell_current);
    }

    void apply_prescribed_magnetic_absorption(double duration, std::vector<double>& energy_change) {
        const std::vector<double> zero(current_operator_->node_count(), 0.0);
        const auto rhs = [&](const std::vector<double>& a) {
            const auto jz = current_operator_->current(a);
            std::vector<double> adot(a.size(), 0.0);
            for (int j = 1; j < c_.ny; ++j) for (int i = 1; i < c_.nx; ++i) {
                const auto n = current_operator_->node_index(i, j);
                const double profile = 0.25 * (absorption_profile_[index(i-1,j-1)] +
                    absorption_profile_[index(i,j-1)] + absorption_profile_[index(i-1,j)] +
                    absorption_profile_[index(i,j)]);
                adot[n] = -c_.absorption_magnetic_diffusivity * profile * jz[n];
            }
            current_operator_->enforce(adot, zero);
            return adot;
        };
        const auto r0 = rhs(potential_);
        auto stage = potential_;
        for (std::size_t n = 0; n < stage.size(); ++n) stage[n] += duration * r0[n];
        const auto r1 = rhs(stage);
        for (std::size_t n = 0; n < stage.size(); ++n)
            potential_[n] = 0.5 * (potential_[n] + stage[n] + duration * r1[n]);
        std::vector<double> old_energy(cells_.size());
        for (std::size_t cell = 0; cell < cells_.size(); ++cell) old_energy[cell] = cells_[cell][total_energy];
        synchronize_current_field(cells_, potential_, true);
        for (std::size_t cell = 0; cell < cells_.size(); ++cell)
            energy_change[cell] += cells_[cell][total_energy] - old_energy[cell];
    }

    void synchronize_current_field(std::vector<State>& data, const std::vector<double>& a,
                                   bool preserve_pressure) const {
        std::vector<double> bxc, byc;
        current_operator_->cell_fields(a, bxc, byc);
        for (std::size_t cell = 0; cell < data.size(); ++cell) {
            if (preserve_pressure) data[cell][total_energy] += 0.5 *
                (bxc[cell] * bxc[cell] + byc[cell] * byc[cell] -
                 data[cell][bx] * data[cell][bx] - data[cell][by] * data[cell][by]);
            data[cell][bx] = bxc[cell]; data[cell][by] = byc[cell];
            data[cell][psi] = 0.0;
        }
    }

    void initialize_prescribed_current(const InitialData& initial) {
        if (periodic_x_ || periodic_y_ || solid_mask_.empty() || !antenna_.enabled())
            throw std::runtime_error("prescribed_current requires nonperiodic boundaries and powered solid cells");
        for (std::size_t cell = 0; cell < cells_.size(); ++cell) {
            if (initial.w[cell] != 0.0 || initial.bz[cell] != 0.0 ||
                std::abs(initial.bx[cell] - initial.bx[0]) > 1e-12 ||
                std::abs(initial.by[cell] - initial.by[0]) > 1e-12)
                throw std::runtime_error("prescribed_current currently supports planar flow and uniform initial Bx,By only");
        }
        if (c_.ambient[4] != 0.0 || c_.ambient[7] != 0.0)
            throw std::runtime_error("prescribed_current requires planar ambient w=Bz=0");
        for (std::size_t n = 0; n < antenna_.jz.size(); ++n) {
            if (antenna_.jx[n] != 0.0 || antenna_.jy[n] != 0.0 ||
                (!is_solid(n % cells_.size()) && antenna_.jz[n] != 0.0))
                throw std::runtime_error("prescribed_current accepts only Jz supported inside solid cells");
        }
        if (antenna_.time.front() > 0.0 || antenna_.time.back() < c_.t_end)
            throw std::runtime_error("prescribed_current table must cover the entire simulation interval");
        current_operator_ = std::make_unique<PrescribedCurrentOperator>(c_.nx, c_.ny, dx_, dy_, solid_mask_);
        if (absorbing_ && c_.absorption_magnetic_diffusivity > 0.0) {
            for (int j = 1; j < c_.ny; ++j) for (int i = 1; i < c_.nx; ++i) {
                if (!current_operator_->powered_mask()[current_operator_->node_index(i, j)]) continue;
                for (int dj = -1; dj <= 1; ++dj) for (int di = -1; di <= 1; ++di)
                    for (int cj = -1; cj <= 0; ++cj) for (int ci = -1; ci <= 0; ++ci)
                        if (absorption_profile_[index(i + di + ci, j + dj + cj)] > 0.0)
                            throw std::runtime_error("Powered body must be separated from the magnetic absorption layer by a free grid layer");
            }
        }
        potential_.resize(current_operator_->node_count());
        for (int j = 0; j <= c_.ny; ++j) for (int i = 0; i <= c_.nx; ++i)
            potential_[current_operator_->node_index(i, j)] = initial.bx[0] * j * dy_ - initial.by[0] * i * dx_;
        current_operator_->enforce(potential_, commanded_current(0.0));
        // Initial nonzero commanded current is an initial, locally screened field.
        // During time integration B and total energy use compatible FV fluxes;
        // pressure-preserving synchronization is used only here and in the sponge.
        synchronize_current_field(cells_, potential_, true);
    }

    void prescribed_induction_fluxes(const std::vector<State>& data, double time,
            std::vector<State>& fx, std::vector<State>& fy,
            std::vector<State>& left, std::vector<State>& bottom, RhsRates& rates) {
        const auto ni = [&](int i, int j) { return current_operator_->node_index(i, j); };
        const auto cell_e = [&](int i, int j) {
            const auto& q = data[index(i, j)];
            return (q[my] * q[bx] - q[mx] * q[by]) / q[rho];
        };
        const auto ex = [&](int i, int j) { // Ez on x face (i,j+1/2)
            i = std::clamp(i, 0, c_.nx); j = std::clamp(j, 0, c_.ny - 1);
            return -(i == 0 ? left[j][by] : fx[index(i - 1, j)][by]);
        };
        const auto ey = [&](int i, int j) { // Ez on y face (i+1/2,j)
            i = std::clamp(i, 0, c_.nx - 1); j = std::clamp(j, 0, c_.ny);
            return j == 0 ? bottom[i][bx] : fy[index(i, j - 1)][bx];
        };
        const auto current = current_operator_->current(*active_potential_);
        potential_rhs_.resize(current_operator_->node_count());
        // E0 corner EMF: preserves the one-dimensional Riemann flux and its
        // numerical dissipation. Physical eta*J is added once on the native grid.
        for (int j = 0; j <= c_.ny; ++j) for (int i = 0; i <= c_.nx; ++i) {
            double ez = 0.5 * (ex(i, j - 1) + ex(i, j) + ey(i - 1, j) + ey(i, j)) -
                0.25 * (cell_e(i - 1, j - 1) + cell_e(i, j - 1) +
                        cell_e(i - 1, j) + cell_e(i, j));
            ez += c_.resistivity * current[ni(i, j)];
            potential_rhs_[ni(i, j)] = -ez;
        }
        // Differentiate the SAME Ampere constraint. This supplies the inductive
        // feed voltage without eta, while every nonpowered node keeps its local
        // plasma Faraday equation. No vacuum field is overwritten in the plasma.
        current_operator_->enforce(potential_rhs_, current_derivative_);
        const auto target = commanded_current(time);
        rates.antenna_power = 0.0;
        for (std::size_t n = 0; n < target.size(); ++n)
            rates.antenna_power += target[n] * potential_rhs_[n] * dx_ * dy_;
        max_stage_current_constraint_error_ = std::max(max_stage_current_constraint_error_,
            current_operator_->max_residual(*active_potential_, target));
        const auto correct = [&](State& flux, int i, int j, int direction) {
            const std::size_t lc = index(i, j), rc = index(i + (direction == 0), j + (direction == 1));
            const Variable tangent = direction == 0 ? by : bx;
            const double bt = 0.5 * (data[lc][tangent] + data[rc][tangent]);
            const double new_flux = direction == 0 ?
                0.5 * (potential_rhs_[ni(i + 1, j)] + potential_rhs_[ni(i + 1, j + 1)]) :
               -0.5 * (potential_rhs_[ni(i, j + 1)] + potential_rhs_[ni(i + 1, j + 1)]);
            // Replace the induction and its Poynting flux together. Total E is
            // otherwise conservative; never add magnetic resync energy as heat.
            flux[total_energy] += bt * (new_flux - flux[tangent]);
            flux[tangent] = new_flux;
            flux[bx + direction] = 0.0;
            flux[bz] = flux[psi] = 0.0;
        };
        for (int j = 0; j < c_.ny; ++j) for (int i = 0; i < c_.nx; ++i) {
            correct(fx[index(i, j)], i, j, 0);
            correct(fy[index(i, j)], i, j, 1);
        }
        for (int j = 0; j < c_.ny; ++j) correct(left[j], -1, j, 0);
        for (int i = 0; i < c_.nx; ++i) correct(bottom[i], i, -1, 1);
    }

    std::vector<State> prescribed_rhs(const std::vector<State>& data, const std::vector<double>& a,
            const std::vector<double>& jdot, double ch, double time, RhsRates& rates,
            std::vector<double>& adot) {
        active_potential_ = &a;
        current_derivative_ = jdot;
        auto rhs = right_hand_side(data, ch, time, rates);
        adot = potential_rhs_;
        active_potential_ = nullptr;
        return rhs;
    }

    void advance_prescribed_current(double dt, double ch, double time) {
        apply_absorption(0.5 * dt);
        const auto j0 = commanded_current(time), j1 = commanded_current(time + dt);
        std::vector<double> jdot(j0.size()), adot0, adot1;
        for (std::size_t n = 0; n < jdot.size(); ++n) jdot[n] = (j1[n] - j0[n]) / dt;
        RhsRates rates0, rates1;
        const auto rhs0 = prescribed_rhs(cells_, potential_, jdot, ch, time, rates0, adot0);
        auto stage_a = potential_;
        for (std::size_t n = 0; n < stage_a.size(); ++n) stage_a[n] += dt * adot0[n];
        std::vector<State> stage(cells_.size());
        std::vector<double> stage_constraint(cells_.size()), final_constraint(cells_.size());
        for (std::size_t cell = 0; cell < cells_.size(); ++cell) {
            for (std::size_t k = 0; k < nvar; ++k) stage[cell][k] = cells_[cell][k] + dt * rhs0[cell][k];
            stage_constraint[cell] = constrain_solid(stage[cell], cell);
            if (!admissible(stage[cell])) throw std::runtime_error("Nonphysical prescribed-current RK stage; lower cfl");
        }
        const auto rhs1 = prescribed_rhs(stage, stage_a, jdot, ch, time + dt, rates1, adot1);
        for (std::size_t n = 0; n < potential_.size(); ++n)
            potential_[n] = 0.5 * (potential_[n] + stage_a[n] + dt * adot1[n]);
        // Remove roundoff drift in the elliptic constraint. This changes only
        // powered nodes at roundoff level, not the exterior plasma solution.
        current_operator_->enforce(potential_, j1);
        for (std::size_t cell = 0; cell < cells_.size(); ++cell) {
            for (std::size_t k = 0; k < nvar; ++k)
                cells_[cell][k] = 0.5 * (cells_[cell][k] + stage[cell][k] + dt * rhs1[cell][k]);
            final_constraint[cell] = constrain_solid(cells_[cell], cell);
            if (!admissible(cells_[cell])) throw std::runtime_error("Nonphysical prescribed-current RK state; lower cfl");
            solid_constraint_energy_ += (0.5 * stage_constraint[cell] + final_constraint[cell]) * dx_ * dy_;
        }
        // Synchronize only floating-point curl drift; no energy is added.
        synchronize_current_field(cells_, potential_, false);
        antenna_energy_injected_ += 0.5 * dt * (rates0.antenna_power + rates1.antenna_power);
        last_circuit_power_ = 0.5 * (rates0.antenna_power + rates1.antenna_power);
        boundary_energy_outflow_ += 0.5 * dt * (rates0.boundary_energy + rates1.boundary_energy);
        boundary_mass_outflow_ += 0.5 * dt * (rates0.boundary_mass + rates1.boundary_mass);
        for (int component = 0; component < 3; ++component) {
            solid_force_[component] = 0.5 * (rates0.solid_force[component] + rates1.solid_force[component]);
            solid_impulse_[component] += dt * solid_force_[component];
        }
        max_wall_mass_flux_ = std::max({max_wall_mass_flux_, rates0.max_wall_mass_flux, rates1.max_wall_mass_flux});
        apply_absorption(0.5 * dt);
    }

    void advance_ssprk2(double dt, double ch, double time) {
        apply_absorption(0.5 * dt);
        RhsRates rates0;
        const std::vector<State> rhs0 = right_hand_side(cells_, ch, time, rates0);
        std::vector<State> stage(cells_.size());
        std::vector<double> stage_constraint, final_constraint;
        if (!solid_mask_.empty()) {
            stage_constraint.resize(cells_.size(), 0.0);
            final_constraint.resize(cells_.size(), 0.0);
        }
        executor_.for_each(cells_.size(), [&](std::size_t cell) {
            for (std::size_t k = 0; k < nvar; ++k) stage[cell][k] = cells_[cell][k] + dt * rhs0[cell][k];
            if (!solid_mask_.empty()) stage_constraint[cell] = constrain_solid(stage[cell], cell);
            if (!admissible(stage[cell])) {
                throw std::runtime_error("State became non-physical in Runge-Kutta stage; lower cfl");
            }
        });
        RhsRates rates1;
        const std::vector<State> rhs1 = right_hand_side(stage, ch, time + dt, rates1);
        executor_.for_each(cells_.size(), [&](std::size_t cell) {
            for (std::size_t k = 0; k < nvar; ++k) {
                cells_[cell][k] = 0.5 * (cells_[cell][k] + stage[cell][k] + dt * rhs1[cell][k]);
            }
            if (!solid_mask_.empty()) final_constraint[cell] = constrain_solid(cells_[cell], cell);
            if (!admissible(cells_[cell])) {
                throw std::runtime_error("State became non-physical after Runge-Kutta step; lower cfl");
            }
        });
        if (c_.glm_damping > 0.0) {
            const double decay = std::exp(-c_.glm_damping * ch * dt / std::min(dx_, dy_));
            executor_.for_each(cells_.size(), [&](std::size_t cell) { cells_[cell][psi] *= decay; });
        }
        antenna_energy_injected_ += 0.5 * dt * (rates0.antenna_power + rates1.antenna_power);
        boundary_energy_outflow_ += 0.5 * dt * (rates0.boundary_energy + rates1.boundary_energy);
        boundary_mass_outflow_ += 0.5 * dt * (rates0.boundary_mass + rates1.boundary_mass);
        if (!solid_mask_.empty()) {
            for (std::size_t cell = 0; cell < cells_.size(); ++cell) {
                solid_constraint_energy_ += (0.5 * stage_constraint[cell] + final_constraint[cell]) * dx_ * dy_;
            }
            for (int component = 0; component < 3; ++component) {
                solid_force_[component] = 0.5 * (rates0.solid_force[component] + rates1.solid_force[component]);
                solid_impulse_[component] += dt * solid_force_[component];
            }
            max_wall_mass_flux_ = std::max({max_wall_mass_flux_, rates0.max_wall_mass_flux, rates1.max_wall_mass_flux});
        }
        apply_absorption(0.5 * dt);
    }

    double derivative(int i, int j, Variable variable, int direction) const {
        if (direction == 0) {
            return (cells_[index(i + 1, j)][variable] - cells_[index(i - 1, j)][variable]) / (2.0 * dx_);
        }
        return (cells_[index(i, j + 1)][variable] - cells_[index(i, j - 1)][variable]) / (2.0 * dy_);
    }

    void write_frame(Hdf5Output& output, int step, double time, double dt) {
        const std::vector<std::string> names = {
            "rho", "u", "v", "w", "p", "Bx", "By", "Bz", "Jx", "Jy", "Jz",
            "Jx_antenna", "Jy_antenna", "Jz_antenna", "divB", "psi"
        };
        std::map<std::string, std::vector<double>> fields;
        for (const std::string& name : names) fields[name].resize(cells_.size());
        auto& out_rho = fields.at("rho"); auto& out_u = fields.at("u");
        auto& out_v = fields.at("v"); auto& out_w = fields.at("w");
        auto& out_p = fields.at("p"); auto& out_bx = fields.at("Bx");
        auto& out_by = fields.at("By"); auto& out_bz = fields.at("Bz");
        auto& out_jx = fields.at("Jx"); auto& out_jy = fields.at("Jy");
        auto& out_jz = fields.at("Jz"); auto& out_divb = fields.at("divB");
        auto& out_jax = fields.at("Jx_antenna"); auto& out_jay = fields.at("Jy_antenna");
        auto& out_jaz = fields.at("Jz_antenna");
        auto& out_psi = fields.at("psi");
        const AntennaSlice antenna = antenna_slice(time);
        std::vector<double> ct_divergence;
        if (current_operator_) {
            std::vector<double> bxf, byf;
            current_operator_->face_fields(potential_, bxf, byf);
            ct_divergence.resize(cells_.size());
            for (int j = 0; j < c_.ny; ++j) for (int i = 0; i < c_.nx; ++i)
                ct_divergence[index(i, j)] =
                    (bxf[static_cast<std::size_t>(j * (c_.nx + 1) + i + 1)] -
                     bxf[static_cast<std::size_t>(j * (c_.nx + 1) + i)]) / dx_ +
                    (byf[static_cast<std::size_t>((j + 1) * c_.nx + i)] -
                     byf[static_cast<std::size_t>(j * c_.nx + i)]) / dy_;
        }
        executor_.for_each(cells_.size(), [&](std::size_t cell) {
            const int i = static_cast<int>(cell % static_cast<std::size_t>(c_.nx));
            const int j = static_cast<int>(cell / static_cast<std::size_t>(c_.nx));
            const Primitive p = primitive(cells_[cell]);
            out_rho[cell] = p.density; out_u[cell] = p.u; out_v[cell] = p.v;
            out_w[cell] = p.w; out_p[cell] = p.pressure;
            out_bx[cell] = p.bx_value; out_by[cell] = p.by_value;
            out_bz[cell] = p.bz_value; out_psi[cell] = p.psi_value;
            out_jx[cell] = derivative(i, j, bz, 1);
            out_jy[cell] = -derivative(i, j, bz, 0);
            out_jz[cell] = derivative(i, j, by, 0) - derivative(i, j, bx, 1);
            const std::array<double, 3> antenna_value = antenna_current(antenna, cell);
            out_jax[cell] = antenna_value[0]; out_jay[cell] = antenna_value[1];
            out_jaz[cell] = antenna_value[2];
            out_divb[cell] = current_operator_ ? ct_divergence[cell] :
                derivative(i, j, bx, 0) + derivative(i, j, by, 1);
        });

        double mass = 0.0, momentum_x = 0.0, momentum_y = 0.0, momentum_z = 0.0;
        double flux_x = 0.0, flux_y = 0.0, flux_z = 0.0, glm = 0.0, sigma = 0.0;
        double energy_integral = 0.0, magnetic = 0.0, kinetic = 0.0, thermal = 0.0, max_speed = 0.0;
        double max_div = 0.0, l1_div = 0.0, max_by = 0.0;
        double antenna_power = 0.0, max_antenna_current = 0.0;
        double min_density = std::numeric_limits<double>::max();
        double min_pressure = std::numeric_limits<double>::max();
        double plasma_mass = 0.0, plasma_energy = 0.0, plasma_magnetic = 0.0;
        double plasma_kinetic = 0.0, plasma_thermal = 0.0, solid_magnetic = 0.0;
        for (std::size_t cell = 0; cell < cells_.size(); ++cell) {
            const Primitive p = primitive(cells_[cell]);
            mass += p.density; momentum_x += cells_[cell][mx]; momentum_y += cells_[cell][my];
            momentum_z += cells_[cell][mz]; flux_x += p.bx_value; flux_y += p.by_value;
            flux_z += p.bz_value; glm += p.psi_value;
            sigma += p.pressure * std::pow(p.density, 1.0 - c_.gamma);
            energy_integral += cells_[cell][total_energy];
            magnetic += 0.5 * (p.bx_value * p.bx_value + p.by_value * p.by_value + p.bz_value * p.bz_value);
            kinetic += 0.5 * p.density * (p.u * p.u + p.v * p.v + p.w * p.w);
            thermal += p.pressure / (c_.gamma - 1.0);
            max_speed = std::max({max_speed, std::abs(p.u) + fast_speed(p, 0),
                                  std::abs(p.v) + fast_speed(p, 1)});
            max_div = std::max(max_div, std::abs(out_divb[cell]));
            l1_div += std::abs(out_divb[cell]);
            max_by = std::max(max_by, std::abs(p.by_value));
            max_antenna_current = std::max({max_antenna_current, std::abs(out_jax[cell]),
                                            std::abs(out_jay[cell]), std::abs(out_jaz[cell])});
            const int i = static_cast<int>(cell % static_cast<std::size_t>(c_.nx));
            const int j = static_cast<int>(cell / static_cast<std::size_t>(c_.nx));
            antenna_power += antenna_source(cells_, i, j, antenna)[total_energy];
            min_density = std::min(min_density, p.density);
            min_pressure = std::min(min_pressure, p.pressure);
            const double magnetic_density = 0.5 *
                (p.bx_value * p.bx_value + p.by_value * p.by_value + p.bz_value * p.bz_value);
            if (is_solid(cell)) {
                solid_magnetic += magnetic_density;
            } else {
                plasma_mass += p.density;
                plasma_energy += cells_[cell][total_energy];
                plasma_magnetic += magnetic_density;
                plasma_kinetic += 0.5 * p.density * (p.u * p.u + p.v * p.v + p.w * p.w);
                plasma_thermal += p.pressure / (c_.gamma - 1.0);
            }
        }
        const double area = dx_ * dy_;
        const double cell_count = static_cast<double>(cells_.size());
        std::map<std::string, double> diagnostics = {
            {"step", static_cast<double>(step)}, {"dt", dt}, {"mass", mass * area},
            {"momentum_x", momentum_x * area}, {"momentum_y", momentum_y * area},
            {"momentum_z", momentum_z * area}, {"magnetic_flux_x", flux_x * area},
            {"magnetic_flux_y", flux_y * area}, {"magnetic_flux_z", flux_z * area},
            {"glm_integral", glm * area}, {"entropy_integral", sigma * area},
            {"total_energy", energy_integral * area},
            {"magnetic_energy", magnetic * area}, {"kinetic_energy", kinetic * area},
            {"thermal_energy", thermal * area}, {"antenna_power", antenna_power * area},
            {"antenna_energy_injected", antenna_energy_injected_},
            {"max_abs_antenna_current", max_antenna_current}, {"max_speed", max_speed},
            {"max_abs_divB", max_div}, {"l1_divB", l1_div / cell_count},
            {"max_abs_By", max_by}, {"min_rho", min_density}, {"min_p", min_pressure},
            {"plasma_mass", plasma_mass * area}, {"plasma_energy", plasma_energy * area},
            {"plasma_magnetic_energy", plasma_magnetic * area},
            {"plasma_kinetic_energy", plasma_kinetic * area},
            {"plasma_thermal_energy", plasma_thermal * area},
            {"solid_magnetic_energy", solid_magnetic * area},
            {"solid_constraint_energy", solid_constraint_energy_},
            {"boundary_energy_outflow", boundary_energy_outflow_},
            {"boundary_mass_outflow", boundary_mass_outflow_},
            {"solid_force_x", solid_force_[0]}, {"solid_force_y", solid_force_[1]},
            {"solid_force_z", solid_force_[2]}, {"solid_impulse_x", solid_impulse_[0]},
            {"solid_impulse_y", solid_impulse_[1]}, {"solid_impulse_z", solid_impulse_[2]},
            {"max_wall_mass_flux", max_wall_mass_flux_},
            {"absorption_energy_exchange", absorption_energy_exchange_},
            {"absorption_mass_exchange", absorption_mass_exchange_},
            {"absorption_momentum_exchange_x", absorption_momentum_exchange_[0]},
            {"absorption_momentum_exchange_y", absorption_momentum_exchange_[1]},
            {"absorption_momentum_exchange_z", absorption_momentum_exchange_[2]}
        };
        std::map<std::string, std::vector<double>> nodes;
        if (current_operator_) {
            const auto target = commanded_current(time);
            nodes["Az"] = potential_;
            nodes["Jz"] = current_operator_->current(potential_);
            nodes["Jz_target"] = target;
            diagnostics["antenna_power"] = last_circuit_power_;
            diagnostics["max_current_constraint_error"] = current_operator_->max_residual(potential_, target);
            diagnostics["max_stage_current_constraint_error"] = max_stage_current_constraint_error_;
            const double native_energy = current_operator_->magnetic_energy(potential_);
            diagnostics["native_magnetic_energy"] = native_energy;
            diagnostics["magnetic_averaging_energy"] = native_energy - magnetic * area;
            diagnostics["current_drive_energy_correction"] = solid_constraint_energy_ - antenna_energy_injected_;
            diagnostics["max_abs_ct_divB"] = max_div;
        }
        output.append(time, fields, diagnostics, nodes);
    }
};

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "Usage: " << argv[0] << " PREPARED_INPUT.h5 OUTPUT.h5\n"
                      << "Normally invoke this through: python3 run.py input/problem.py\n";
            return 2;
        }
        SolverInput input = read_solver_input(argv[1]);
        Solver solver(std::move(input.config), std::move(input.initial), std::move(input.antenna),
                      std::move(input.solid_mask));
        solver.run(argv[2]);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

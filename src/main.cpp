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
    H5Gclose(group);

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
    H5Fclose(file);

    if (c.nx < 4 || c.ny < 4 || c.lx <= 0.0 || c.ly <= 0.0) {
        throw std::runtime_error("The grid must have at least 4 cells per direction and positive lengths");
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
    return {c, std::move(initial), std::move(antenna)};
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
               int threads_used)
        : nx_(c.nx), ny_(c.ny) {
        const std::filesystem::path path(filename);
        if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
        file_ = h5_id(H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
                      "creating output file " + filename);
        write_attribute<int>(file_, "format_version", 3);
        write_string_attribute(file_, "model", "dimensionless 2.5-D driven resistive GLM-MHD");
        write_attribute<int>(file_, "nx", c.nx); write_attribute<int>(file_, "ny", c.ny);
        write_attribute<double>(file_, "lx", c.lx); write_attribute<double>(file_, "ly", c.ly);
        write_attribute<double>(file_, "gamma", c.gamma);
        write_attribute<double>(file_, "resistivity", c.resistivity);
        write_attribute<int>(file_, "threads_used", threads_used);
        write_attribute<int>(file_, "antenna_enabled", antenna.enabled() ? 1 : 0);
        const double antenna_sample_interval = antenna.sample_count() > 1
            ? antenna.time[1] - antenna.time[0] : 0.0;
        write_attribute<double>(file_, "antenna_sample_interval", antenna_sample_interval);

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
    }

    Hdf5Output(const Hdf5Output&) = delete;
    Hdf5Output& operator=(const Hdf5Output&) = delete;

    ~Hdf5Output() {
        for (auto& item : field_datasets_) H5Dclose(item.second);
        for (auto& item : diagnostic_datasets_) H5Dclose(item.second);
        if (time_dataset_ >= 0) H5Dclose(time_dataset_);
        if (file_ >= 0) H5Fclose(file_);
    }

    void append(double time, const std::map<std::string, std::vector<double>>& fields,
                const std::map<std::string, double>& diagnostics) {
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
            "max_speed", "max_abs_divB", "l1_divB", "max_abs_By", "min_rho", "min_p"
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
    Solver(Config config, InitialData initial, AntennaData antenna)
        : c_(std::move(config)), dx_(c_.lx / c_.nx), dy_(c_.ly / c_.ny),
          cells_(static_cast<std::size_t>(c_.nx * c_.ny)), antenna_(std::move(antenna)),
          executor_(c_.requested_threads) {
        initialize(initial);
    }

    void run(const std::string& output_filename) {
        Hdf5Output output(output_filename, c_, antenna_, executor_.thread_count());
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
            if (!(dt > 0.0) || !std::isfinite(dt)) throw std::runtime_error("Invalid time step");

            advance_ssprk2(dt, ch, time);
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
    std::vector<State> cells_;
    AntennaData antenna_;
    ParallelExecutor executor_;
    double antenna_energy_injected_ = 0.0;

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
        return static_cast<std::size_t>(wrapped(j, c_.ny) * c_.nx + wrapped(i, c_.nx));
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
        const State& center = data[index(i, j)];
        const State& lower = data[index(i - (direction == 0), j - (direction == 1))];
        const State& upper = data[index(i + (direction == 0), j + (direction == 1))];
        for (std::size_t k = 0; k < nvar; ++k) {
            const double backward = center[k] - lower[k];
            const double forward = upper[k] - center[k];
            result[k] = minmod(c_.limiter_theta * backward, 0.5 * (backward + forward),
                               c_.limiter_theta * forward);
        }
        return result;
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

    State face_flux(const std::vector<State>& data, const std::vector<State>& slopes,
                    int i, int j, int direction, double ch) const {
        const std::size_t left_index = index(i, j);
        const std::size_t right_index = index(i + (direction == 0), j + (direction == 1));
        State left = data[left_index], right = data[right_index];
        for (std::size_t k = 0; k < nvar; ++k) {
            left[k] += 0.5 * slopes[left_index][k];
            right[k] -= 0.5 * slopes[right_index][k];
        }
        if (!admissible(left)) left = data[left_index];
        if (!admissible(right)) right = data[right_index];
        State flux = hlld_flux(left, right, direction, ch);
        if (c_.resistivity > 0.0) {
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
                                       double time, double& antenna_power) {
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

        const AntennaSlice antenna = antenna_slice(time);
        std::vector<State> rhs(data.size());
        std::vector<double> antenna_power_density(
            antenna.active ? data.size() : 0, 0.0);
        executor_.for_each(data.size(), [&](std::size_t cell) {
            const int i = static_cast<int>(cell % static_cast<std::size_t>(c_.nx));
            const int j = static_cast<int>(cell / static_cast<std::size_t>(c_.nx));
            const State& left_flux = flux_x[index(i - 1, j)];
            const State& lower_flux = flux_y[index(i, j - 1)];
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
        antenna_power = 0.0;
        for (const double power_density : antenna_power_density) antenna_power += power_density;
        antenna_power *= dx_ * dy_;
        return rhs;
    }

    double cleaning_speed(const std::vector<State>& data) const {
        double maximum = 0.0;
        for (const State& q : data) {
            const Primitive p = primitive(q);
            maximum = std::max({maximum, std::abs(p.u) + fast_speed(p, 0),
                                std::abs(p.v) + fast_speed(p, 1)});
        }
        return c_.glm_speed_factor * std::max(maximum, 1.0e-12);
    }

    double stable_timestep(const std::vector<State>& data, double ch) const {
        double speed_x = ch, speed_y = ch;
        for (const State& q : data) {
            const Primitive p = primitive(q);
            speed_x = std::max(speed_x, std::abs(p.u) + fast_speed(p, 0));
            speed_y = std::max(speed_y, std::abs(p.v) + fast_speed(p, 1));
        }
        double dt = c_.cfl / (speed_x / dx_ + speed_y / dy_);
        if (c_.resistivity > 0.0) {
            const double diffusion_dt = c_.diffusion_cfl /
                (c_.resistivity * (1.0 / (dx_ * dx_) + 1.0 / (dy_ * dy_)));
            dt = std::min(dt, diffusion_dt);
        }
        return dt;
    }

    void advance_ssprk2(double dt, double ch, double time) {
        double antenna_power0 = 0.0;
        const std::vector<State> rhs0 = right_hand_side(cells_, ch, time, antenna_power0);
        std::vector<State> stage(cells_.size());
        executor_.for_each(cells_.size(), [&](std::size_t cell) {
            for (std::size_t k = 0; k < nvar; ++k) stage[cell][k] = cells_[cell][k] + dt * rhs0[cell][k];
            if (!admissible(stage[cell])) {
                throw std::runtime_error("State became non-physical in Runge-Kutta stage; lower cfl");
            }
        });
        double antenna_power1 = 0.0;
        const std::vector<State> rhs1 = right_hand_side(stage, ch, time + dt, antenna_power1);
        executor_.for_each(cells_.size(), [&](std::size_t cell) {
            for (std::size_t k = 0; k < nvar; ++k) {
                cells_[cell][k] = 0.5 * (cells_[cell][k] + stage[cell][k] + dt * rhs1[cell][k]);
            }
            if (!admissible(cells_[cell])) {
                throw std::runtime_error("State became non-physical after Runge-Kutta step; lower cfl");
            }
        });
        if (c_.glm_damping > 0.0) {
            const double decay = std::exp(-c_.glm_damping * ch * dt / std::min(dx_, dy_));
            executor_.for_each(cells_.size(), [&](std::size_t cell) { cells_[cell][psi] *= decay; });
        }
        antenna_energy_injected_ += 0.5 * dt * (antenna_power0 + antenna_power1);
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
            out_divb[cell] = derivative(i, j, bx, 0) + derivative(i, j, by, 1);
        });

        double mass = 0.0, momentum_x = 0.0, momentum_y = 0.0, momentum_z = 0.0;
        double flux_x = 0.0, flux_y = 0.0, flux_z = 0.0, glm = 0.0, sigma = 0.0;
        double energy_integral = 0.0, magnetic = 0.0, kinetic = 0.0, thermal = 0.0, max_speed = 0.0;
        double max_div = 0.0, l1_div = 0.0, max_by = 0.0;
        double antenna_power = 0.0, max_antenna_current = 0.0;
        double min_density = std::numeric_limits<double>::max();
        double min_pressure = std::numeric_limits<double>::max();
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
        }
        const double area = dx_ * dy_;
        const double cell_count = static_cast<double>(cells_.size());
        const std::map<std::string, double> diagnostics = {
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
            {"max_abs_By", max_by}, {"min_rho", min_density}, {"min_p", min_pressure}
        };
        output.append(time, fields, diagnostics);
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
        Solver solver(std::move(input.config), std::move(input.initial), std::move(input.antenna));
        solver.run(argv[2]);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

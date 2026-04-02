// Nanobind Python bindings for the C++ eikonal ray tracer.
//
// Provides:
//   - EikonalTracer (type-erased, accepts Python callables or table)
//   - trace(), trace_batch(), trace_batch_full()
//   - speed_from_eta() helper
//   - geodetic_to_ecef(), ecef_to_geodetic()

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/function.h>

#include <refraction/ray_trace_batch.hpp>
#include <refraction/geodetic.hpp>

namespace nb = nanobind;
using namespace nb::literals;
using namespace refraction;

// Helper: flatten a vector of Vec3 into a flat double vector
static std::vector<double> flatten_vec3(const std::vector<Vec3>& vecs) {
    std::vector<double> flat(vecs.size() * 3);
    for (size_t i = 0; i < vecs.size(); ++i) {
        flat[3*i]   = vecs[i][0];
        flat[3*i+1] = vecs[i][1];
        flat[3*i+2] = vecs[i][2];
    }
    return flat;
}

// Helper: pack an EikonalResult into a Python dict
static nb::dict result_to_dict(const EikonalResult& res) {
    nb::dict d;
    d["s"] = nb::cast(res.s);
    d["T"] = nb::cast(res.T);
    d["v"] = nb::cast(res.v);
    d["h_m"] = nb::cast(res.h_m);
    d["lat_deg"] = nb::cast(res.lat_deg);
    d["lon_deg"] = nb::cast(res.lon_deg);
    d["theta_deg"] = nb::cast(res.theta_deg);
    d["azimuth_deg"] = nb::cast(res.azimuth_deg);
    d["curvature"] = nb::cast(res.curvature);
    d["dh_ds"] = nb::cast(res.dh_ds);
    d["dv_ds"] = nb::cast(res.dv_ds);
    d["momentum_error"] = nb::cast(res.momentum_error);
    d["bending_deg"] = res.bending_deg;
    d["terminated_by"] = res.terminated_by;

    // Vec3 arrays as flat vectors (Python reshapes to Nx3)
    d["r"] = nb::cast(flatten_vec3(res.r));
    d["p"] = nb::cast(flatten_vec3(res.p));
    d["r_prime"] = nb::cast(flatten_vec3(res.r_prime));
    d["p_prime"] = nb::cast(flatten_vec3(res.p_prime));
    d["r_double_prime"] = nb::cast(flatten_vec3(res.r_double_prime));
    d["dr_dT"] = nb::cast(flatten_vec3(res.dr_dT));
    d["d2r_dT2"] = nb::cast(flatten_vec3(res.d2r_dT2));
    d["dp_dT"] = nb::cast(flatten_vec3(res.dp_dT));

    return d;
}

NB_MODULE(refraction_native, m) {
    m.doc() = "C++ eikonal ray tracer with batch support (nanobind)";

    // --- Geodetic utilities ---
    m.def("geodetic_to_ecef", [](double lat, double lon, double h) {
        auto r = geodetic_to_ecef(lat, lon, h);
        return nb::make_tuple(r[0], r[1], r[2]);
    }, "lat_deg"_a, "lon_deg"_a, "h_m"_a);

    m.def("ecef_to_geodetic", [](double X, double Y, double Z) {
        auto g = ecef_to_geodetic(X, Y, Z);
        return nb::make_tuple(g.lat_deg, g.lon_deg, g.h_m);
    }, "X"_a, "Y"_a, "Z"_a);

    // --- speed_from_eta ---
    m.def("speed_from_eta", [](nb::callable eta, nb::callable deta, double c_ref) {
        SpeedFunc v = [eta, c_ref](double h) {
            return c_ref / nb::cast<double>(eta(h));
        };
        SpeedFunc dv = [eta, deta, c_ref](double h) {
            double e = nb::cast<double>(eta(h));
            return -c_ref * nb::cast<double>(deta(h)) / (e * e);
        };
        return nb::make_tuple(v, dv);
    }, "eta_func"_a, "deta_dh_func"_a, "c_ref"_a);

    // --- TableSpeedProfile ---
    nb::class_<TableSpeedProfile>(m, "TableSpeedProfile")
        .def("__init__", [](TableSpeedProfile* self,
                            nb::ndarray<double, nb::ndim<1>> h_arr,
                            nb::ndarray<double, nb::ndim<1>> v_arr) {
            int n = static_cast<int>(h_arr.shape(0));
            new (self) TableSpeedProfile(h_arr.data(), v_arr.data(), n);
        }, "h_array"_a, "v_array"_a,
        "Construct from altitude and speed arrays (must be sorted by altitude)")
        .def("speed", &TableSpeedProfile::speed)
        .def("dspeed", &TableSpeedProfile::dspeed);

    // --- EikonalTracer ---
    nb::class_<EikonalTracerErased>(m, "EikonalTracer")
        .def("__init__", [](EikonalTracerErased* self,
                            nb::callable v_func, nb::callable dv_func) {
            SpeedFunc v = [v_func](double h) {
                return nb::cast<double>(v_func(h));
            };
            SpeedFunc dv = [dv_func](double h) {
                return nb::cast<double>(dv_func(h));
            };
            new (self) EikonalTracerErased(std::move(v), std::move(dv));
        }, "v_func"_a, "dv_dh_func"_a,
        "Create tracer from Python speed callables")

        .def_static("from_table", [](const TableSpeedProfile& table) {
            return EikonalTracerErased(table);
        }, "table"_a,
        "Create tracer from a TableSpeedProfile (pure C++, no Python callbacks)")

        .def("trace", [](const EikonalTracerErased& self,
                         double lat, double lon, double h,
                         double elev, double az,
                         std::optional<double> target_alt,
                         std::optional<double> target_time,
                         double max_arc, bool ground_hit,
                         double rtol, double atol, double max_step) {
            EikonalInput inp{lat, lon, h, elev, az};
            EikonalStopCondition stop{target_alt, target_time, max_arc, ground_hit};
            auto res = self.trace(inp, stop, rtol, atol, max_step);
            return result_to_dict(res);
        },
        "lat_deg"_a, "lon_deg"_a, "h_m"_a,
        "elevation_deg"_a, "azimuth_deg"_a,
        "target_altitude_m"_a = nb::none(),
        "target_travel_time_s"_a = nb::none(),
        "max_arc_length_m"_a = 500000.0,
        "detect_ground_hit"_a = true,
        "rtol"_a = 1e-10, "atol"_a = 1e-12, "max_step"_a = 100.0)

        .def("trace_batch", [](const EikonalTracerErased& self,
                                nb::ndarray<double, nb::ndim<2>> inputs,
                                double rtol, double atol, double max_step) {
            int n = static_cast<int>(inputs.shape(0));
            std::vector<BatchInput> batch(n);
            for (int i = 0; i < n; ++i) {
                batch[i].lat_deg = inputs(i, 0);
                batch[i].lon_deg = inputs(i, 1);
                batch[i].h_m = inputs(i, 2);
                batch[i].elevation_deg = inputs(i, 3);
                batch[i].azimuth_deg = inputs(i, 4);
                batch[i].travel_time_s = inputs(i, 5);
            }

            auto results = trace_batch(self, batch, rtol, atol, max_step);

            // Pack all endpoint fields
            auto flat3 = [&](auto getter) {
                std::vector<double> f(n * 3);
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < 3; ++j)
                        f[3*i+j] = getter(results[i])[j];
                return f;
            };
            auto arr1 = [&](auto getter) {
                std::vector<double> a(n);
                for (int i = 0; i < n; ++i) a[i] = getter(results[i]);
                return a;
            };

            nb::dict d;
            d["r"] = nb::cast(flat3([](const BatchOutput& o) { return o.r; }));
            d["p"] = nb::cast(flat3([](const BatchOutput& o) { return o.p; }));
            d["r_prime"] = nb::cast(flat3([](const BatchOutput& o) { return o.r_prime; }));
            d["p_prime"] = nb::cast(flat3([](const BatchOutput& o) { return o.p_prime; }));
            d["r_double_prime"] = nb::cast(flat3([](const BatchOutput& o) { return o.r_double_prime; }));
            d["dr_dT"] = nb::cast(flat3([](const BatchOutput& o) { return o.dr_dT; }));
            d["d2r_dT2"] = nb::cast(flat3([](const BatchOutput& o) { return o.d2r_dT2; }));
            d["dp_dT"] = nb::cast(flat3([](const BatchOutput& o) { return o.dp_dT; }));

            d["s"] = nb::cast(arr1([](const BatchOutput& o) { return o.s; }));
            d["T"] = nb::cast(arr1([](const BatchOutput& o) { return o.T; }));
            d["v"] = nb::cast(arr1([](const BatchOutput& o) { return o.v; }));
            d["dh_ds"] = nb::cast(arr1([](const BatchOutput& o) { return o.dh_ds; }));
            d["dv_ds"] = nb::cast(arr1([](const BatchOutput& o) { return o.dv_ds; }));
            d["curvature"] = nb::cast(arr1([](const BatchOutput& o) { return o.curvature; }));
            d["lat_deg"] = nb::cast(arr1([](const BatchOutput& o) { return o.lat_deg; }));
            d["lon_deg"] = nb::cast(arr1([](const BatchOutput& o) { return o.lon_deg; }));
            d["h_m"] = nb::cast(arr1([](const BatchOutput& o) { return o.h_m; }));
            d["theta_deg"] = nb::cast(arr1([](const BatchOutput& o) { return o.theta_deg; }));
            d["azimuth_deg"] = nb::cast(arr1([](const BatchOutput& o) { return o.azimuth_deg; }));
            d["bending_deg"] = nb::cast(arr1([](const BatchOutput& o) { return o.bending_deg; }));
            d["momentum_error"] = nb::cast(arr1([](const BatchOutput& o) { return o.momentum_error; }));
            d["n"] = n;
            return d;
        },
        "inputs"_a, "rtol"_a = 1e-10, "atol"_a = 1e-12, "max_step"_a = 100.0,
        "Trace N rays (endpoint only). inputs: (N,6) array [lat,lon,h,elev,az,travel_time]")

        .def("trace_batch_full", [](const EikonalTracerErased& self,
                                     nb::ndarray<double, nb::ndim<2>> inputs,
                                     double rtol, double atol, double max_step) {
            int n = static_cast<int>(inputs.shape(0));
            std::vector<BatchInput> batch(n);
            for (int i = 0; i < n; ++i) {
                batch[i].lat_deg = inputs(i, 0);
                batch[i].lon_deg = inputs(i, 1);
                batch[i].h_m = inputs(i, 2);
                batch[i].elevation_deg = inputs(i, 3);
                batch[i].azimuth_deg = inputs(i, 4);
                batch[i].travel_time_s = inputs(i, 5);
            }

            auto results = trace_batch_full(self, batch, rtol, atol, max_step);

            // Return list of dicts, one per ray (full path)
            nb::list out;
            for (const auto& res : results) {
                out.append(result_to_dict(res));
            }
            return out;
        },
        "inputs"_a, "rtol"_a = 1e-10, "atol"_a = 1e-12, "max_step"_a = 100.0,
        "Trace N rays (full path). inputs: (N,6) array [lat,lon,h,elev,az,travel_time]");
}

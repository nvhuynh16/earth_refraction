#pragma once

// Shared test fixtures for atmospheric refraction tests.

#include <refraction/atmosphere/nrlmsis21.hpp>

namespace refraction {
namespace test {

inline atmosphere::NRLMSIS21::Input standard_input() {
    atmosphere::NRLMSIS21::Input inp;
    inp.day = 172;      // Summer solstice
    inp.utsec = 29000;  // ~8 UT
    inp.lat = 45.0;
    inp.lon = -75.0;
    inp.f107a = 150.0;
    inp.f107 = 150.0;
    inp.ap = {4, 4, 4, 4, 4, 4, 4};
    return inp;
}

}  // namespace test
}  // namespace refraction

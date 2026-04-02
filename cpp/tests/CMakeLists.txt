set(TEST_SOURCES
    test_nrlmsis.cpp
    test_water_vapor.cpp
    test_ciddor.cpp
    test_itu.cpp
    test_species.cpp
    test_surface_anchor.cpp
    test_profile.cpp
    test_meissner_wentz.cpp
    test_millard_seaver.cpp
    test_ocean_profile.cpp
    test_ocean_refraction.cpp
    test_del_grosso.cpp
    test_chen_millero.cpp
    test_mackenzie.cpp
    test_sound_speed.cpp
    test_geodetic.cpp
    test_ray_trace.cpp
)

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE refraction)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

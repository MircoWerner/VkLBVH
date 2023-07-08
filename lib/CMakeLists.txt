### SPIRV-Reflect ###
add_library(spirv-reflect SHARED ${CMAKE_CURRENT_SOURCE_DIR}/SPIRV-Reflect/spirv_reflect.h ${CMAKE_CURRENT_SOURCE_DIR}/SPIRV-Reflect/spirv_reflect.cpp)
set_target_properties(spirv-reflect PROPERTIES LINKER_LANGUAGE CXX)
#target_include_directories(spirv-reflect PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS spirv-reflect LIBRARY DESTINATION lib)

### tinyobjloader ###
add_library(tinyobjloader SHARED ${CMAKE_CURRENT_SOURCE_DIR}/tinyobjloader/tiny_obj_loader.h)
set_target_properties(tinyobjloader PROPERTIES LINKER_LANGUAGE CXX)
#target_include_directories(tinyobjloader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS tinyobjloader LIBRARY DESTINATION lib)
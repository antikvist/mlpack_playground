option(ENABLE_BOOST "Enable boost" ${ENABLE_LIBRARIES})

if (NOT ENABLE_BOOST)
    message(STATUS "Not using boost")
    return()
endif()

set (LIBRARY_DIR "${CMAKE_SOURCE_DIR}/contrib/boost")

set (SRCS_SERIALIZATION
    "${LIBRARY_DIR}/libs/serialization/src/archive_exception.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/basic_archive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/basic_iarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/basic_iserializer.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/basic_oarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/basic_oserializer.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/basic_pointer_iserializer.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/basic_pointer_oserializer.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/basic_serializer_map.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/basic_text_iprimitive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/basic_text_oprimitive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/basic_text_wiprimitive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/basic_text_woprimitive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/basic_xml_archive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/basic_xml_grammar.ipp"
    "${LIBRARY_DIR}/libs/serialization/src/binary_iarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/binary_oarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/binary_wiarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/binary_woarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/codecvt_null.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/extended_type_info_no_rtti.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/extended_type_info_typeid.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/extended_type_info.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/polymorphic_binary_iarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/polymorphic_iarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/polymorphic_oarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/polymorphic_text_iarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/polymorphic_text_oarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/polymorphic_text_wiarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/polymorphic_text_woarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/polymorphic_xml_iarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/polymorphic_xml_oarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/polymorphic_xml_wiarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/polymorphic_xml_woarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/stl_port.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/text_iarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/text_oarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/text_wiarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/text_woarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/utf8_codecvt_facet.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/void_cast.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/xml_archive_exception.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/xml_grammar.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/xml_iarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/xml_oarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/xml_wgrammar.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/xml_wiarchive.cpp"
    "${LIBRARY_DIR}/libs/serialization/src/xml_woarchive.cpp"
)

add_library (_boost_serialization ${SRCS_SERIALIZATION})
add_library (ch_contib::boost_serialization ALIAS _boost_serialization)
target_include_directories (_boost_serialization PRIVATE ${LIBRARY_DIR})
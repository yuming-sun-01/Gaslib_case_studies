# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:46:57 2024

@author: ssnaik
"""
import re
import pandas as pd
import xml.etree.ElementTree as ET

def get_compressor_coefficients(file_path):
    def extract_xml_from_csharp(file_path):
        with open(file_path, 'r') as file:
            csharp_code = file.read()
            match = re.search(r'(?s)<compressorStations.*?</compressorStations>', csharp_code)
            if match:
                return match.group(0)
            else:
                raise ValueError("XML data not found in the C# source file.")

    xml_data = extract_xml_from_csharp(file_path)
    # Parse the XML data
    tree = ET.ElementTree(ET.fromstring(xml_data))
    root = tree.getroot()
    
    # Define the namespace
    namespace = {'ns': 'http://gaslib.zib.de/CompressorStations'}
    
    # Extract coefficients
    coefficients = []
    for compressor_station in root.findall('.//ns:compressorStation', namespace):
        turbo_compressor = compressor_station.find('.//ns:turboCompressor', namespace)
        if turbo_compressor is not None:
            n_isoline_coeffs = [float(turbo_compressor.find(f'.//ns:n_isoline_coeff_{i}', namespace).attrib['value']) for i in range(1, 10)]
            eta_ad_isoline_coeffs = [float(turbo_compressor.find(f'.//ns:eta_ad_isoline_coeff_{i}', namespace).attrib['value']) for i in range(1, 10)]
            surgeline_coeffs = [float(turbo_compressor.find(f'.//ns:surgeline_coeff_{i}', namespace).attrib['value']) for i in range(1, 4)]
            chokeline_coeffs = [float(turbo_compressor.find(f'.//ns:chokeline_coeff_{i}', namespace).attrib['value']) for i in range(1, 4)]
            
            coefficients.append({
                'compressor_station_id': compressor_station.attrib['id'],
                'n_isoline_coeffs': n_isoline_coeffs,
                'eta_ad_isoline_coeffs': eta_ad_isoline_coeffs,
                'surgeline_coeffs': surgeline_coeffs,
                'chokeline_coeffs': chokeline_coeffs
            })
    
    coef_n_isoline, coef_n_eta, coef_surge, coef_choke = make_data_frame(coefficients)
    return coef_n_isoline, coef_n_eta, coef_surge, coef_choke

def make_data_frame(coefficients):
    coef_n_isoline = []
    coef_n_eta = []
    coef_surge = []
    coef_choke = []    
    for item in coefficients:
        coef_n_isoline.append([item['compressor_station_id'],
                             *item['n_isoline_coeffs']])
        coef_n_eta.append([item['compressor_station_id'],
                             *item['eta_ad_isoline_coeffs']])
        coef_surge.append([item['compressor_station_id'],
                             *item['surgeline_coeffs']])
        coef_choke.append([item['compressor_station_id'],
                             *item['chokeline_coeffs']])
    return coef_n_isoline, coef_n_eta, coef_surge, coef_choke
    
if __name__ == "__main__":
    import os
    # Path to the C# source file
    csharp_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data_files", "GasLib_11", "compressor_coefs.cs",
    )
    
    # Extract coefficients from XML data in C# source file
    coefficients = get_compressor_coefficients(csharp_file_path)

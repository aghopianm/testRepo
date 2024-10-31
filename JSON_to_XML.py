import json
import xmltodict

def json_to_xml(json_file, xml_file):
    # Read JSON data from the file
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    # Convert JSON to XML
    xml_data = xmltodict.unparse({"converted data": json_data}, pretty=True)

    # Write the XML data to a file
    with open(xml_file, 'w') as file:
        file.write(xml_data)

# Specify the input JSON file and output XML file
json_file = 'People.json'
xml_file = 'PeopleXML.xml'

# Call the conversion function
json_to_xml(json_file, xml_file)

print(f"Converted {json_file} to {xml_file}")
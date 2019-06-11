/* tslint:disable */

import * as THREE from 'three';

/**
 * @author Wei Meng / http://about.me/menway
 *
 * Description: A THREE loader for PLY ASCII files (known as the Polygon
 * File Format or the Stanford Triangle Format).
 *
 * Limitations: ASCII decoding assumes file is UTF-8.
 *
 * Usage:
 *  let loader = new THREE.PLYLoader();
 *  loader.load('./models/ply/ascii/dolphins.ply', function (geometry) {
 *
 *		scene.add( new THREE.Mesh( geometry ) );
 *
 *	} );
 *
 * If the PLY file uses non standard property names, they can be mapped while
 * loading. For example, the following maps the properties
 * “diffuse_(red|green|blue)” in the file to standard color names.
 *
 * loader.setPropertyNameMapping( {
 *	diffuse_red: 'red',
 *	diffuse_green: 'green',
 *	diffuse_blue: 'blue'
 * } );
 *
 */

interface PropertyType {
  name: string;
  countType: string;
  itemType: string;
  type: string;
}

interface ElementType {
  name: string;
  count: number;
  properties: PropertyType[];
}

interface HeaderType {
  comments: string[];
  elements: ElementType[];
  headerLength: number;
  format: string;
  version: string;
}

interface BufferType {
  indices: number[];
  vertices: number[];
  normals: number[];
  uvs: number[];
  colors: number[];
}

export class PLYLoader {
  private manager: THREE.LoadingManager;
  private propertyNameMapping: Record<string, string>;

  constructor(manager?: THREE.LoadingManager) {
    this.manager = (manager !== undefined) ?
        manager :
        THREE.DefaultLoadingManager;

    this.propertyNameMapping = {};
  }

  public load(url: string, onLoad: (geometry: THREE.BufferGeometry) => void,
              onProgress: (xhr: ProgressEvent) => void,
              onError: () => void) {

    const scope = this;

    const loader = new THREE.FileLoader(this.manager);
    loader.setResponseType('arraybuffer');
    loader.load(url, function(text) {

      onLoad(scope.parse(text));

    }, onProgress, onError);

  }

  public setPropertyNameMapping(mapping: Record<string, string>) {
    this.propertyNameMapping = mapping;
  }

  private parse(data: string | ArrayBuffer) {

    function parseHeader(data: string | ArrayBuffer) {

      const patternHeader = /ply([\s\S]*)end_header\s/;
      let headerText = '';
      let headerLength = 0;
      const result = patternHeader.exec(data as string);

      if (result !== null) {

        headerText = result[1];
        headerLength = result[0].length;

      }

      const header: HeaderType = {
        comments: [],
        elements: [],
        headerLength,
        format: '',
        version: ''
      };

      const lines = headerText.split('\n');
      let currentElement: ElementType = {
        name: '',
        count: 0,
        properties: []
      };
      let lineType;
      let lineValues;

      function make_ply_element_property(propertValues: string[],
                                         propertyNameMapping: Record<string,
                                                                     string>):
        PropertyType {
        const property: PropertyType = {
          name: '',
          type: propertValues[0],
          itemType: '',
          countType: ''
        };

        if (property.type === 'list') {
          property.name = propertValues[3];
          property.countType = propertValues[1];
          property.itemType = propertValues[2];
        } else {
          property.name = propertValues[1];
        }

        if (property.name in propertyNameMapping) {
          property.name = propertyNameMapping[property.name];
        }

        return property;
      }

      for (let line of lines) {
        line = line.trim();

        if (line === '') {
          continue;
        }

        lineValues = line.split(/\s+/);
        lineType = lineValues.shift();
        line = lineValues.join(' ');

        switch (lineType) {
          case 'format':
            header.format = lineValues[0];
            header.version = lineValues[1];
            break;

          case 'comment':
            header.comments.push(line);
            break;

          case 'element':
            if (currentElement !== undefined) {
              header.elements.push(currentElement);
            }

            currentElement = {
              name: lineValues[0],
              count: parseInt(lineValues[1], 10),
              properties: []
            };
            break;

          case 'property':
            currentElement.properties.push(make_ply_element_property(lineValues,
                scope.propertyNameMapping));
            break;

          default:

        }
      }

      if (currentElement !== undefined) {
        header.elements.push(currentElement);
      }

      return header;
    }

    function parseASCIINumber(n: string, type: string): number {
      switch (type) {
        case 'char':
        case 'uchar':
        case 'short':
        case 'ushort':
        case 'int':
        case 'uint':
        case 'int8':
        case 'uint8':
        case 'int16':
        case 'uint16':
        case 'int32':
        case 'uint32':
          return parseInt(n, 10);
        case 'float':
        case 'double':
        case 'float32':
        case 'float64':
          return parseFloat(n);
        default:
          return 0;
      }
    }

    function parseASCIIElement(properties: PropertyType[], line: string) {
      const values = line.split(/\s+/);
      const element: Record<string, number | number[]> = {};

      for (const property of properties) {
        if (property.type === 'list') {
          const list: number[] = [];
          let nextToken = values.shift();

          if (nextToken) {
            const n = parseASCIINumber(nextToken, property.countType);

            for (let j = 0; j < n; j++) {
              nextToken = values.shift();
              if (nextToken) {
                list.push(parseASCIINumber(nextToken, property.itemType));
              }
            }
            element[property.name] = list;
          }
        } else {
          const token = values.shift();
          if (token) {
            element[property.name] = parseASCIINumber(token,
              property.type);
          }
        }
      }
      return element;

    }

    function parseASCII(data: string, header: HeaderType):
      THREE.BufferGeometry {

      // PLY ascii format specification, as per http://en.wikipedia.org/wiki/PLY_(file_format)

      const buffer: BufferType = {
        indices: [],
        vertices: [],
        normals: [],
        uvs: [],
        colors: []
      };

      const patternBody = /end_header\s([\s\S]*)$/;
      let body = '';
      const result = patternBody.exec(data);

      if (result) {
        body = result[1];
      }

      const lines = body.split('\n');
      let currentElement = 0;
      let currentElementCount = 0;

      for (let line of lines) {
        line = line.trim();
        if (line === '') {
          continue;
        }

        if (currentElementCount >= header.elements[currentElement].count) {
          currentElement++;
          currentElementCount = 0;
        }

        const element = parseASCIIElement(
            header.elements[currentElement].properties, line);

        handleElement(buffer, header.elements[currentElement].name, element);

        currentElementCount++;

      }

      return postProcess(buffer);

    }

    function postProcess(buffer: BufferType) {

      const geometry = new THREE.BufferGeometry();

      // mandatory buffer data

      if (buffer.indices.length > 0) {

        geometry.setIndex(buffer.indices);

      }

      geometry.addAttribute('position',
          new THREE.Float32BufferAttribute(buffer.vertices, 3));

      // optional buffer data

      if (buffer.normals.length > 0) {

        geometry.addAttribute('normal',
            new THREE.Float32BufferAttribute(buffer.normals, 3));

      }

      if (buffer.uvs.length > 0) {

        geometry.addAttribute('uv',
            new THREE.Float32BufferAttribute(buffer.uvs, 2));

      }

      if (buffer.colors.length > 0) {
        geometry.addAttribute('color',
            new THREE.Float32BufferAttribute(buffer.colors, 3));
      }

      geometry.computeBoundingSphere();
      return geometry;
    }

    function handleElement(buffer: BufferType, elementName: string,
                           element: Record<string, number | number[]>) {

      /* tslint:disable:no-string-literal */
      if (elementName === 'vertex') {
        buffer.vertices.push(
          element['x'] as number,
          element['y'] as number,
          element['z'] as number
        );

        if ('nx' in element && 'ny' in element && 'nz' in element) {
          buffer.normals.push(
            element['nx'] as number,
            element['ny'] as number,
            element['nz'] as number
          );
        }

        if ('s' in element && 't' in element) {
          buffer.uvs.push(element['s'] as number, element['t'] as number);
        }

        if ('red' in element && 'green' in element && 'blue' in element) {
          buffer.colors.push(
            (element['red'] as number) / 255.0,
            (element['green'] as number) / 255.0,
            (element['blue'] as number) / 255.0
          );
        }
      } else if (elementName === 'face') {

        const vertexIndices =
          (element.vertex_indices || element.vertex_index) as number[];

        if (vertexIndices.length === 3) {

          buffer.indices.push(vertexIndices[0], vertexIndices[1],
              vertexIndices[2]);

        } else if (vertexIndices.length === 4) {

          buffer.indices.push(vertexIndices[0], vertexIndices[1],
              vertexIndices[3]);
          buffer.indices.push(vertexIndices[1], vertexIndices[2],
              vertexIndices[3]);
        }
      }
    }

    function binaryRead(dataview: DataView, at: number, type: string,
                        littleEndian: boolean) {

      switch (type) {
        case 'int8':
        case 'char':
          return [dataview.getInt8(at), 1];
        case 'uint8':
        case 'uchar':
          return [dataview.getUint8(at), 1];
        case 'int16':
        case 'short':
          return [dataview.getInt16(at, littleEndian), 2];
        case 'uint16':
        case 'ushort':
          return [dataview.getUint16(at, littleEndian), 2];
        case 'int32':
        case 'int':
          return [dataview.getInt32(at, littleEndian), 4];
        case 'uint32':
        case 'uint':
          return [dataview.getUint32(at, littleEndian), 4];
        case 'float32':
        case 'float':
          return [dataview.getFloat32(at, littleEndian), 4];
        case 'float64':
        case 'double':
          return [dataview.getFloat64(at, littleEndian), 8];
        default:
          return [0, 0];
      }

    }

    function binaryReadElement(dataview: DataView, at: number,
                               properties: PropertyType[],
                               littleEndian: boolean):
      [Record<string, number | number[]>, number] {

      const element: Record<string, number | number[]> = {};
      let result: number[];
      let read = 0;

      for (const property of properties) {
        if (property.type === 'list') {
          const list: number[] = [];

          result = binaryRead(dataview, at + read, property.countType,
              littleEndian);
          const n = result[0];
          read += result[1];

          for (let j = 0; j < n; j++) {

            result = binaryRead(dataview, at + read, property.itemType,
                littleEndian);
            list.push(result[0]);
            read += result[1];

          }

          element[property.name] = list;
        } else {
          result = binaryRead(dataview, at + read, property.type,
              littleEndian);
          element[property.name] = result[0];
          read += result[1];
        }
      }
      return [element, read];
    }

    function parseBinary(data: ArrayBuffer, header: HeaderType) {
      const buffer: BufferType = {
        indices: [],
        vertices: [],
        normals: [],
        uvs: [],
        colors: []
      };

      const littleEndian = (header.format === 'binary_little_endian');
      const body = new DataView(data, header.headerLength);
      let result: [Record<string, number | number[]>, number];
      let loc = 0;

      for (const currentElement of header.elements) {

        for (let currentElementCount = 0; currentElementCount <
             currentElement.count; currentElementCount++) {

          result = binaryReadElement(body, loc,
              currentElement.properties, littleEndian);
          loc += result[1];
          const element = result[0];

          handleElement(buffer, currentElement.name, element);

        }

      }

      return postProcess(buffer);

    }

    //

    let geometry;
    const scope = this;

    if (data instanceof ArrayBuffer) {
      const text = THREE.LoaderUtils.decodeText(new Uint8Array(data));
      const header = parseHeader(text);

      geometry = header.format === 'ascii' ?
          parseASCII(text, header) :
          parseBinary(data, header);

    } else {

      geometry = parseASCII(data, parseHeader(data));

    }

    return geometry;

  }

}

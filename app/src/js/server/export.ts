import _ from 'lodash'
import { AttributeToolType, LabelTypeName } from '../common/types'
import { PointType } from '../drawable/2d/path_point2d'
import { Attribute, ConfigType,
  ItemType, Node2DType, PolygonType, State
} from '../functional/types'
import { transformBox2D, transformBox3D, transformPlane3D } from './bdd_type_transformers'
import { ItemExport, LabelExport, PolygonExportType } from './bdd_types'

/**
 * Converts a polygon label to export format
 */
export function convertPolygonToExport (
  poly2d: PolygonType, labelType: string): PolygonExportType[] {
  const typeCharacters = poly2d.points.map(
    (point) => {
      switch (point.pointType) {
        case PointType.CURVE:
          return 'C'
        case PointType.VERTEX:
          return 'L'
      }

      return ''
    }
  )
  const types = typeCharacters.join('')
  const vertices: Array<[number, number]> =
    poly2d.points.map((point) => [point.x, point.y])
  return [{
    vertices,
    types,
    closed: labelType === LabelTypeName.POLYGON_2D
  }]
}

/**
 * converts single item to exportable format
 * @param config
 * @param item
 */
export function convertItemToExport (
  config: ConfigType,
  item: ItemType
): ItemExport[] {
  const itemExports: {[sensor: number]: ItemExport} = {}
  for (const key of Object.keys(item.urls)) {
    const sensor = Number(key)
    const url = item.urls[sensor]
    const videoName = item.videoName
    const timestamp = (item.timestamp) ? item.timestamp : 0
    itemExports[sensor] = {
      name: url,
      url,
      videoName,
      timestamp,
      attributes: {},
      labels: [],
      sensor
    }
  }
  // TODO: Clean up the export code for naming and modularity
  for (const key of Object.keys(item.labels)) {
    const label = item.labels[key]
    const labelExport: LabelExport = {
      id: label.id,
      category: config.categories[label.category[0]],
      attributes: parseLabelAttributes(label.attributes, config.attributes),
      manualShape: label.manual,
      box2d: null,
      poly2d: null,
      box3d: null,
      plane3d: null,
      customs: {}
    }
    if (label.shapes.length > 0) {
      const shapeId0 = label.shapes[0]
      const shape0 = item.shapes[shapeId0]
      switch (label.type) {
        case LabelTypeName.BOX_2D:
          labelExport.box2d = transformBox2D(shape0)
          break
        case LabelTypeName.POLYGON_2D:
        case LabelTypeName.POLYLINE_2D:
          labelExport.poly2d = convertPolygonToExport(
            shape0 as PolygonType, label.type
          )
          break
        case LabelTypeName.BOX_3D:
          labelExport.box3d = transformBox3D(shape0)
          break
        case LabelTypeName.PLANE_3D:
          labelExport.plane3d = transformPlane3D(shape0)
          break
        default:
          if (label.type in config.label2DTemplates) {
            const points: Array<[number ,number]> = []
            const names: string[] = []
            const hidden: boolean[] = []
            for (const shapeId of label.shapes) {
              const node = item.shapes[shapeId] as Node2DType
              points.push([node.x, node.y])
              names.push(node.name)
              hidden.push(Boolean(node.hidden))
            }

            const template = config.label2DTemplates[label.type]

            labelExport.customs[template.name] = {
              points, names, hidden, edges: template.edges
            }
          }
      }
    }
    for (const sensor of label.sensors) {
      itemExports[sensor].labels.push(labelExport)
    }
  }
  return Object.values(itemExports)
}

/**
 * parses attributes into BDD format
 * @param attributes
 */
function parseLabelAttributes (labelAttributes: {[key: number]: number[]},
                               configAttributes: Attribute[]):
  {[key: string]: (string[] | boolean) } {
  const exportAttributes: {[key: string]: (string[] | boolean) } = {}
  Object.entries(labelAttributes).forEach(([key, attributeList]) => {
    const index = parseInt(key, 10)
    const attribute = configAttributes[index]
    if (attribute.toolType === AttributeToolType.LIST
        || attribute.toolType === AttributeToolType.LONG_LIST) {
      // List attribute case- check whether each value is applied
      const selectedValues: string[] = []
      attributeList.forEach((valueIndex) => {
        if (valueIndex in attribute.values) {
          selectedValues.push(attribute.values[valueIndex])
        }
      })
      exportAttributes[attribute.name] = selectedValues
    } else if (attribute.toolType === AttributeToolType.SWITCH) {
      // Boolean attribute case- should be a single boolean in the list
      let value = false
      if (attributeList.length > 0) {
        const attributeVal = attributeList[0]
        if (attributeVal === 1) {
          value = true
        }
      }
      exportAttributes[attribute.name] = value
    }

  })
  return exportAttributes
}

/**
 * converts state to export format
 * @param state
 */
export function convertStateToExport (state: State)
: ItemExport[] {
  const config = state.task.config
  const items = state.task.items
  const exportList: ItemExport[] = []
  items.forEach((item) => {
    exportList.push(...convertItemToExport(config, item))
  })
  return exportList
}

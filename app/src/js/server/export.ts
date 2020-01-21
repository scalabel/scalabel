import { AttributeToolType, LabelTypeName } from '../common/types'
// import { PathPoint2D, PointType } from '../drawable/2d/path_point2d'
import { ItemExport, LabelExport } from '../functional/bdd_types'
import { Attribute, ConfigType, CubeType,
  ItemType, Node2DType, RectType, State } from '../functional/types'

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
    const label = item.labels[Number(key)]
    const labelExport: LabelExport = {
      id: label.id,
      category: config.categories[label.category[0]],
      attributes: parseLabelAttributes(label.attributes, config.attributes),
      manualShape: label.manual,
      box2d: null,
      poly2d: null,
      box3d: null,
      customs: {}
    }
    if (label.shapes.length > 0) {
      const firstShapeId = label.shapes[0]
      const firstIndexedShape = item.shapes[firstShapeId]
      switch (label.type) {
        case LabelTypeName.BOX_2D:
          const box2d = firstIndexedShape.shape as RectType
          labelExport.box2d = box2d
          break
        // case LabelTypeName.POLYGON_2D:
        // case LabelTypeName.POLYLINE_2D:
        //   const poly2d = firstIndexedShape.shape as PathPoint2D[]
        //   const typeCharacters = poly2d.points.map(
        //     (point) => {
        //       switch (point.type) {
        //         case PointType.CURVE:
        //           return 'C'
        //         case PointType.VERTEX:
        //           return 'L'
        //       }

        //       return ''
        //     }
        //   )
        //   const types = typeCharacters.join('')
        //   const vertices: Array<[number, number]> =
        //     poly2d.points.map((point) => [point.x, point.y])
        //   labelExport.poly2d = [{
        //     vertices,
        //     types,
        //     closed: label.type === LabelTypeName.POLYGON_2D
        //   }]
        //   break
        case LabelTypeName.BOX_3D:
          const poly3d = firstIndexedShape.shape as CubeType
          labelExport.box3d = poly3d
          break
        default:
          if (label.type in config.label2DTemplates) {
            const points: Array<[number ,number]> = []
            const names: string[] = []
            const hidden: boolean[] = []
            for (const shapeId of label.shapes) {
              const node = item.shapes[shapeId].shape as Node2DType
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
      // list attribute case- check whether each value is applied
      const selectedValues: string[] = []
      attributeList.forEach((valueIndex) => {
        if (valueIndex in attribute.values) {
          selectedValues.push(attribute.values[valueIndex])
        }
      })
      exportAttributes[attribute.name] = selectedValues
    } else if (attribute.toolType === AttributeToolType.SWITCH) {
      // boolean attribute case- should be a single boolean in the list
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

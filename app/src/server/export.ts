import { AttributeToolType, LabelTypeName } from "../const/common"
import { isValidId } from "../functional/states"
import { ItemExport, LabelExport, PolygonExportType } from "../types/export"
import {
  Attribute,
  ConfigType,
  IdType,
  ItemType,
  Node2DType,
  PathPoint2DType,
  PathPointType,
  State
} from "../types/state"
import {
  extrinsicsToExport,
  intrinsicsToExport,
  transformBox2D,
  transformBox3D,
  transformPlane3D
} from "./bdd_type_transformers"
import { getChildLabelIds } from "../functional/common"

/**
 * Converts a polygon label to export format
 *
 * @param pathPoints
 * @param labelType
 */
export function convertPolygonToExport(
  pathPoints: PathPoint2DType[],
  labelType: string
): PolygonExportType {
  const typeCharacters = pathPoints.map((point) => {
    switch (point.pointType) {
      case PathPointType.CURVE:
        return "C"
      case PathPointType.LINE:
        return "L"
    }

    return ""
  })
  const types = typeCharacters.join("")
  const vertices: Array<[number, number]> = pathPoints.map((point) => [
    point.x,
    point.y
  ])
  return {
    vertices,
    types,
    closed: labelType === LabelTypeName.POLYGON_2D
  }
}

/**
 * converts single item to exportable format
 *
 * @param config
 * @param item
 */
export function convertItemToExport(
  config: ConfigType,
  item: ItemType
): ItemExport[] {
  const itemExports: { [sensor: number]: ItemExport } = {}
  for (const key of Object.keys(item.urls)) {
    const sensor = Number(key)
    const url = item.urls[sensor]
    const videoName = item.videoName
    const timestamp = item.timestamp
    let name = url
    if (item.names !== undefined && Object.keys(item.names).includes(key)) {
      name = item.names[sensor]
    }
    itemExports[sensor] = {
      name: name,
      url,
      videoName,
      timestamp,
      attributes: {},
      labels: [],
      sensor,
      intrinsics:
        item.intrinsics !== undefined
          ? intrinsicsToExport(item.intrinsics[sensor])
          : undefined,
      extrinsics:
        item.extrinsics !== undefined
          ? extrinsicsToExport(item.extrinsics[sensor])
          : undefined
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
      box3d: null
    }

    // if it is not a parent label, ignore it
    let labelIds: IdType[] = []
    if (isValidId(label.parent)) {
      continue
    } else {
      labelIds = getChildLabelIds(item, key)
    }
    if (label.shapes.length > 0 || labelIds.length > 0) {
      const shapeId0 = label.shapes[0]
      const shape0 = item.shapes[shapeId0]
      switch (label.type) {
        case LabelTypeName.BOX_2D:
          labelExport.box2d = transformBox2D(shape0)
          break
        case LabelTypeName.POLYGON_2D:
        case LabelTypeName.POLYLINE_2D:
        case LabelTypeName.EMPTY:
          labelExport.poly2d = labelIds.map((labelId) => {
            const childLabel = item.labels[labelId]
            return convertPolygonToExport(
              childLabel.shapes.map((s) => item.shapes[s]) as PathPoint2DType[],
              childLabel.type
            )
          })
          break
        case LabelTypeName.BOX_3D:
          labelExport.box3d = transformBox3D(shape0)
          break
        case LabelTypeName.PLANE_3D:
          labelExport.plane3d = transformPlane3D(shape0)
          break
        default:
          if (label.type in config.label2DTemplates) {
            const points: Array<[number, number]> = []
            const names: string[] = []
            const hidden: boolean[] = []
            for (const shapeId of label.shapes) {
              const node = item.shapes[shapeId] as Node2DType
              points.push([node.x, node.y])
              names.push(node.name)
              hidden.push(Boolean(node.hidden))
            }
          }
      }
    }
    if (isValidId(label.track)) {
      // If the label is in a track, use id of the track as label id
      labelExport.id = label.track
    }
    for (const sensor of label.sensors) {
      if (label.type === LabelTypeName.TAG) {
        itemExports[sensor].attributes = parseLabelAttributes(
          label.attributes,
          config.attributes
        ) as { [key: string]: string | string[] }
      } else {
        itemExports[sensor].labels.push(labelExport)
      }
    }
  }
  return Object.values(itemExports)
}

/**
 * parses attributes into Scalabel format
 *
 * @param attributes
 * @param configAttributes
 */
function parseLabelAttributes(
  labelAttributes: { [key: number]: number[] },
  configAttributes: Attribute[]
): { [key: string]: string | string[] | boolean } {
  const exportAttributes: { [key: string]: string | string[] | boolean } = {}
  Object.entries(labelAttributes).forEach(([key, attributeList]) => {
    const index = parseInt(key, 10)
    const attribute = configAttributes[index]
    if (
      attribute.type === AttributeToolType.LIST ||
      attribute.type === AttributeToolType.LONG_LIST
    ) {
      // List attribute case- check whether each value is applied
      const selectedValues: string[] = []
      attributeList.forEach((valueIndex) => {
        if (valueIndex in attribute.values) {
          selectedValues.push(attribute.values[valueIndex])
        }
      })
      if (selectedValues.length === 1) {
        exportAttributes[attribute.name] = selectedValues[0]
      } else if (selectedValues.length > 1) {
        exportAttributes[attribute.name] = selectedValues
      }
    } else if (attribute.type === AttributeToolType.SWITCH) {
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
 *
 * @param state
 */
export function convertStateToExport(state: State): ItemExport[] {
  const config = state.task.config
  const items = state.task.items
  const exportList: ItemExport[] = []
  items.forEach((item) => {
    exportList.push(...convertItemToExport(config, item))
  })
  return exportList
}

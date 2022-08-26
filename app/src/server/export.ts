import { AttributeToolType, LabelTypeName } from "../const/common"
import { isValidId } from "../functional/states"
import { ItemExport, LabelExport, PolygonExportType } from "../types/export"
import {
  Attribute,
  ConfigType,
  ItemType,
  LabelIdMap,
  LabelType,
  Node2DType,
  PathPoint2DType,
  PathPointType,
  ShapeIdMap,
  State
} from "../types/state"
import {
  extrinsicsToExport,
  intrinsicsToExport,
  transformBox2D,
  transformBox3D,
  transformPlane3D
} from "./bdd_type_transformers"

/**
 * Converts a polygon label to export format
 *
 * @param pathPoints
 * @param labelType
 */
export function convertPolygonToExport(
  pathPoints: PathPoint2DType[],
  labelType: string
): PolygonExportType[] {
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
  return [
    {
      vertices,
      types,
      closed: labelType === LabelTypeName.POLYGON_2D
    }
  ]
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

    // `intrinsics` and `extrinsics` can somehow be `{"-1": null}`, which
    // contradicts the definition of `ItemType` interface, unfortunately, thus
    // care must be taken when checking the validity.
    const intrinsics = item.intrinsics?.[sensor] ?? undefined
    const extrinsics = item.extrinsics?.[sensor] ?? undefined

    itemExports[sensor] = {
      name,
      url,
      videoName,
      timestamp,
      attributes: {},
      labels: [],
      sensor,
      intrinsics:
        intrinsics !== undefined ? intrinsicsToExport(intrinsics) : undefined,
      extrinsics:
        extrinsics !== undefined ? extrinsicsToExport(extrinsics) : undefined
    }
  }

  const labels = convertLabelsToExport(item.labels, item.shapes, config)
  labels.forEach((l) => {
    const l0 = item.labels[l.id]

    if (isValidId(l0.parent)) {
      return
    }

    for (const sensor of l0.sensors) {
      if (l0.type === LabelTypeName.TAG) {
        itemExports[sensor].attributes = parseLabelAttributes(
          l0.attributes,
          config.attributes
        ) as { [key: string]: string | string[] }
      } else {
        // Extrernally, all internal differnet labels of the same track should
        // share a common id, which is conveniently set to the id of the track.
        if (isValidId(l0.track)) {
          l.id = l0.track
        }
        itemExports[sensor].labels.push(l)
      }
    }
  })

  return Object.values(itemExports)
}

/**
 * converts a list of labels to exportable format
 *
 * @param labelMap
 * @param shapeMap
 * @param config
 */
export function convertLabelsToExport(
  labelMap: LabelIdMap,
  shapeMap: ShapeIdMap,
  config: ConfigType
): LabelExport[] {
  const polyTypes = [LabelTypeName.POLYGON_2D, LabelTypeName.POLYLINE_2D]
  const isPolygon = polyTypes.some((p) => config.labelTypes.includes(p))
  const fn = isPolygon
    ? convertPolygonLabelsToExport
    : convertNonPolygonLabelsToExport
  return fn(labelMap, shapeMap, config)
}

/**
 * converts a list of non-polygon labels to exportable format
 *
 * @param labelMap
 * @param shapeMap
 * @param config
 */
export function convertNonPolygonLabelsToExport(
  labelMap: LabelIdMap,
  shapeMap: ShapeIdMap,
  config: ConfigType
): LabelExport[] {
  return Object.entries(labelMap).map(([_id, l]) =>
    convertNonPolygonLabelToExport(l, shapeMap, config)
  )
}

/**
 * converts a single non-polygon label to exportable format
 *
 * @param label
 * @param shapeMap
 * @param config
 */
export function convertNonPolygonLabelToExport(
  label: LabelType,
  shapeMap: ShapeIdMap,
  config: ConfigType
): LabelExport {
  const labelExport: LabelExport = {
    id: label.id,
    category: config.categories[label.category[0]],
    attributes: parseLabelAttributes(label.attributes, config.attributes),
    manualShape: label.manual,
    box2d: null,
    poly2d: null,
    box3d: null
  }

  if (label.shapes.length === 0) {
    return labelExport
  }

  const shapeId0 = label.shapes[0]
  const shape0 = shapeMap[shapeId0]
  switch (label.type) {
    case LabelTypeName.BOX_2D:
      labelExport.box2d = transformBox2D(shape0)
      break
    case LabelTypeName.POLYGON_2D:
    case LabelTypeName.POLYLINE_2D:
      throw new Error("unexpectedly found polygon shape")
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
          const node = shapeMap[shapeId] as Node2DType
          points.push([node.x, node.y])
          names.push(node.name)
          hidden.push(Boolean(node.hidden))
        }
      }
  }

  return labelExport
}

/**
 * converts a list of polygon labels to exportable format
 *
 * @param labelMap
 * @param shapeMap
 * @param config
 */
export function convertPolygonLabelsToExport(
  labelMap: LabelIdMap,
  shapeMap: ShapeIdMap,
  config: ConfigType
): LabelExport[] {
  // All linked labels will be exported to a single label, with its polygon
  // array filled with all polygons of these labels.

  // Key is the root id of the tree of link labels.
  const polygons = new Map<string, PolygonExportType[]>()

  Object.entries(labelMap).forEach(([_id, l]) => {
    const pts = l.shapes.map((sid) => shapeMap[sid]) as PathPoint2DType[]
    if (pts.length === 0) {
      return
    }

    const ps = convertPolygonToExport(pts, l.type)

    // Find the root of the tree to which this label belongs
    while (isValidId(l.parent)) {
      l = labelMap[l.parent]
    }

    // Append polygons to the array of the root
    const rid = l.id
    if (!polygons.has(rid)) {
      polygons.set(rid, [])
    }
    polygons.get(rid)?.push(...ps)
  })

  const labels = Array.from(polygons).map(([rid, ps]) => {
    const root = labelMap[rid]
    return {
      id: root.id,
      category: config.categories[root.category[0]],
      attributes: parseLabelAttributes(root.attributes, config.attributes),
      manualShape: root.manual,
      poly2d: ps,
      box2d: null,
      box3d: null
    }
  })

  return labels
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

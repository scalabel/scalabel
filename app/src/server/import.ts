import _ from "lodash"

import { AttributeToolType, LabelTypeName } from "../const/common"
import {
  makeCube,
  makeItem,
  makeLabel,
  makeLabelExport,
  makePathPoint2D,
  makePlane,
  makeRect
} from "../functional/states"
import {
  ExtrinsicsExportType,
  IntrinsicsExportType,
  ItemExport,
  LabelExport
} from "../types/export"
import {
  Attribute,
  ExtrinsicsType,
  IdType,
  IntrinsicsType,
  ItemType,
  LabelIdMap,
  LabelType,
  PathPointType,
  ShapeIdMap,
  ShapeType
} from "../types/state"
import { uid } from "../common/uid"
import {
  box3dToCube,
  extrinsicsFromExport,
  intrinsicsFromExport
} from "./bdd_type_transformers"

/**
 * Convert the attributes to label to ensure the format consistency
 *
 * @param attributes
 */
function convertAttributeToLabel(attributes: {
  [key: string]: string | string[]
}): LabelExport {
  return makeLabelExport({
    attributes
  })
}

/**
 * Converts single exported item to frontend state format
 *
 * @param item the item in export format
 * @param videoName
 * @param timestamp
 * @param itemIndex the item index (relative to task)
 * @param itemId the item id (relative to project)
 * @param attributesNameMap look up an attribute and its index from its name
 * @param attributeValueMap look up an attribute value's index
 * @param categoryNameMap look up a category's index from its name
 * @param tracking
 * @param labelTypes
 */
export function convertItemToImport(
  videoName: string,
  timestamp: number,
  itemExportMap: { [id: number]: Partial<ItemExport> },
  itemIndex: number,
  itemId: number,
  attributeNameMap: { [key: string]: [number, Attribute] },
  attributeValueMap: { [key: string]: number },
  categoryNameMap: { [key: string]: number },
  tracking: boolean,
  labelTypes: string[]
): ItemType {
  const urls: { [id: number]: string } = {}
  const names: { [id: number]: string } = {}
  const intrinsics: { [id: number]: IntrinsicsType } = {}
  const extrinsics: { [id: number]: ExtrinsicsType } = {}

  const labels: LabelIdMap = {}
  const shapes: ShapeIdMap = {}
  for (const key of Object.keys(itemExportMap)) {
    const sensorId = Number(key)
    urls[sensorId] = itemExportMap[sensorId].url as string
    if (itemExportMap[sensorId].name !== undefined) {
      names[sensorId] = itemExportMap[sensorId].name as string
    }
    if (
      itemExportMap[sensorId].intrinsics !== undefined &&
      itemExportMap[sensorId].intrinsics !== null
    ) {
      intrinsics[sensorId] = intrinsicsFromExport(
        itemExportMap[sensorId].intrinsics as IntrinsicsExportType
      )
    }
    if (
      itemExportMap[sensorId].extrinsics !== undefined &&
      itemExportMap[sensorId].extrinsics !== null
    ) {
      extrinsics[sensorId] = extrinsicsFromExport(
        itemExportMap[sensorId].extrinsics as ExtrinsicsExportType
      )
    }
    let labelsExport = itemExportMap[sensorId].labels
    const itemAttributes = itemExportMap[sensorId].attributes
    let isTagging = false
    if (
      itemAttributes !== undefined &&
      itemAttributes !== null &&
      Object.keys(itemAttributes).length > 0
    ) {
      if (labelsExport === undefined) {
        labelsExport = []
      }
      labelsExport = labelsExport.concat(
        convertAttributeToLabel(itemAttributes)
      )
      isTagging = true
    }
    if (labelsExport !== undefined) {
      for (const labelExport of labelsExport) {
        let labelId = labelExport.id.toString()
        if (tracking) {
          labelId = uid()
        }
        // Each label may appear in multiple sensors
        if (tracking && labelExport.id in labels) {
          labels[labelId].sensors.push(sensorId)
          continue
        }

        const categories: number[] = []
        if (labelExport.category in categoryNameMap) {
          categories.push(categoryNameMap[labelExport.category])
        }

        let attributes: { [key: number]: number[] } = {}
        if (
          labelExport.attributes !== undefined &&
          labelExport.attributes !== null
        ) {
          attributes = parseExportAttributes(
            labelExport.attributes,
            attributeNameMap,
            attributeValueMap
          )
        }

        const [importedLabels, importedShapes] = convertLabelToImport(
          labelExport,
          itemIndex,
          sensorId,
          labelTypes,
          categories,
          attributes,
          labelId
        )

        importedLabels.forEach((l) => {
          if (isTagging) {
            l.type = "tag"
          }

          if (tracking) {
            l.track = labelExport.id.toString()
          }

          labels[l.id] = l
        })

        for (const indexedShape of importedShapes) {
          shapes[indexedShape.id] = indexedShape
        }
      }
    }
  }

  return makeItem(
    {
      names,
      urls,
      index: itemIndex,
      id: itemId.toString(),
      timestamp,
      videoName,
      labels,
      shapes,
      intrinsics: Object.keys(intrinsics).length > 0 ? intrinsics : undefined,
      extrinsics: Object.keys(extrinsics).length > 0 ? extrinsics : undefined
    },
    true
  )
}

/**
 * parses attributes from Scalabel format (strings)
 * to internal format (index in config's attributes)
 *
 * @param attributesExport the attributes to process
 * @param attributesNameMap look up an attribute and its index from its name
 * @param attributeValueMap look up an attribute value's index
 */
function parseExportAttributes(
  attributesExport: { [key: string]: string | string[] | boolean },
  attributeNameMap: { [key: string]: [number, Attribute] },
  attributeValueMap: { [key: string]: number }
): { [key: number]: number[] } {
  const labelAttributes: { [key: number]: number[] } = {}
  Object.entries(attributesExport).forEach(([name, attributeList]) => {
    // Get the config attribute that matches the exported attribute name
    if (typeof attributeList === "string") {
      attributeList = [attributeList]
    }
    if (name in attributeNameMap) {
      const [configIndex, currentAttribute] = attributeNameMap[name]
      // Load the attribute based on its type
      if (currentAttribute.type === AttributeToolType.SWITCH) {
        // Boolean attribute case- only two choices, not a list
        let value = 0
        const attributeVal = attributeList as boolean
        if (attributeVal) {
          value = 1
        }
        labelAttributes[configIndex] = [value]
      } else if (
        currentAttribute.type === AttributeToolType.LIST ||
        currentAttribute.type === AttributeToolType.LONG_LIST
      ) {
        // List attribute case- can choose multiple values
        const selectedIndices: number[] = []
        const attributeValues = attributeList as string[]
        attributeValues.forEach((value: string) => {
          // Get the index of the selected value
          const valueIndex = attributeValueMap[value]
          if (valueIndex !== -1) {
            selectedIndices.push(valueIndex)
          }
        })
        labelAttributes[configIndex] = selectedIndices
      }
    }
  })
  return labelAttributes
}

/**
 * based on the label in export format, create a label in internal format
 * and update the corresponding shapes in the map
 *
 * @param label the label in export format
 * @param shapesImport map to update, from shapeId to shape
 * @param labelExport
 * @param item
 * @param sensorId
 * @param category
 * @param attributes
 * @param labelTypes
 * @param newLabelId
 */
function convertLabelToImport(
  labelExport: LabelExport,
  item: number,
  sensorId: number,
  labelTypes: string[],
  category?: number[],
  attributes?: { [key: number]: number[] },
  newLabelId?: string
): [LabelType[], ShapeType[]] {
  const polyTypes = [LabelTypeName.POLYGON_2D, LabelTypeName.POLYLINE_2D]
  if (polyTypes.some((p) => labelTypes.includes(p))) {
    // Unfortunately, importing labels with more than one polygons were not
    // supported before. To avoid potentially breaking change, we handle this case
    // separately.
    if ((labelExport.poly2d?.length ?? 0) > 1) {
      return convertPolygonLabelToImport(
        labelExport,
        item,
        sensorId,
        category,
        attributes
      )
    }
  }

  let labelType = LabelTypeName.EMPTY
  let shapes: null | ShapeType[] = null
  let labelId = labelExport.id.toString()
  if (newLabelId !== undefined) {
    labelId = newLabelId
  }

  /**
   * Convert each import shape based on their type
   */
  if (
    labelTypes.includes(LabelTypeName.BOX_2D) &&
    labelExport.box2d !== null &&
    labelExport.box2d !== undefined
  ) {
    labelType = LabelTypeName.BOX_2D
    shapes = [makeRect(labelExport.box2d)]
  } else if (
    (labelTypes.includes(LabelTypeName.POLYGON_2D) ||
      labelTypes.includes(LabelTypeName.POLYLINE_2D)) &&
    labelExport.poly2d !== null &&
    labelExport.poly2d !== undefined
  ) {
    const polyExport = labelExport.poly2d[0]
    labelType = polyExport.closed
      ? LabelTypeName.POLYGON_2D
      : LabelTypeName.POLYLINE_2D
    shapes = polyExport.vertices.map((vertex, i) =>
      makePathPoint2D({
        x: vertex[0],
        y: vertex[1],
        pointType:
          polyExport.types[i] === "L" ? PathPointType.LINE : PathPointType.CURVE
      })
    )
  } else if (
    labelTypes.includes(LabelTypeName.BOX_3D) &&
    labelExport.box3d !== null &&
    labelExport.box3d !== undefined
  ) {
    labelType = LabelTypeName.BOX_3D
    shapes = [makeCube(box3dToCube(labelExport.box3d))]
  } else if (
    labelTypes.includes(LabelTypeName.BOX_3D) &&
    labelExport.plane3d !== null &&
    labelExport.plane3d !== undefined
  ) {
    labelType = LabelTypeName.PLANE_3D
    shapes = [makePlane(labelExport.plane3d)]
  }

  // If the label has any shapes, import them too
  const shapeIds: IdType[] = []
  const shapeImports: ShapeType[] = []
  if (shapes !== null) {
    _.forEach(shapes, (s) => {
      s.label.push(labelId)
      shapeIds.push(s.id)
      shapeImports.push(s)
    })
  }

  const labelImport = makeLabel(
    {
      id: labelId,
      type: labelType,
      item,
      shapes: shapeIds,
      manual: labelExport.manualShape || false,
      category,
      attributes,
      sensors: [sensorId]
    },
    false
  )

  return [[labelImport], shapeImports]
}

/**
 * convert an external polygon to internal label(s)
 *
 * @param labelExport
 * @param item
 * @param sensorId
 * @param category
 */
function convertPolygonLabelToImport(
  labelExport: LabelExport,
  item: number,
  sensorId: number,
  category?: number[],
  attributes?: { [key: number]: number[] }
): [LabelType[], ShapeType[]] {
  const { poly2d: polygons, manualShape: manual } = labelExport
  if (polygons == null) {
    return [[], []]
  }

  // Make code shorter to fit in one line.
  const LT = LabelTypeName
  const PT = PathPointType

  const rootId = uid()

  const labels: LabelType[] = []
  const shapes: ShapeType[] = []
  polygons.forEach((p) => {
    const lid = uid()

    const ltype = p.closed ? LT.POLYGON_2D : LT.POLYLINE_2D
    const points = p.vertices.map(([x, y], i) => {
      const pointType = p.types[i] === "L" ? PT.LINE : PT.CURVE
      const point = makePathPoint2D({ x, y, pointType })
      point.label.push(lid)
      return point
    })

    const pids = points.map((p) => p.id)
    const label = makeLabel(
      {
        id: lid,
        type: ltype,
        item,
        shapes: pids,
        manual,
        category,
        attributes,
        sensors: [sensorId],
        parent: rootId
      },
      false
    )

    labels.push(label)
    shapes.push(...points)
  })

  const root = makeLabel(
    {
      id: rootId,
      item,
      manual,
      category,
      children: labels.map((l) => l.id),

      // The data structure design does not make it clear how the `type` of
      // the root label of a unioned label should be. Nevertheless, we set it to
      // `POLYGON_2D` to make everything work.
      type: LT.POLYGON_2D
    },
    false
  )

  return [[root, ...labels], shapes]
}

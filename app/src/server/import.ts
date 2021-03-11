import _ from "lodash"

import { AttributeToolType, LabelTypeName } from "../const/common"
import {
  makeCube,
  makeItem,
  makeLabel,
  makePathPoint2D,
  makePlane,
  makeRect
} from "../functional/states"
import { ItemExport, LabelExport } from "../types/export"
import {
  Attribute,
  IdType,
  ItemType,
  LabelIdMap,
  LabelType,
  PathPointType,
  ShapeIdMap,
  ShapeType
} from "../types/state"

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
  tracking: boolean
): ItemType {
  const urls: { [id: number]: string } = {}

  const labels: LabelIdMap = {}
  const shapes: ShapeIdMap = {}
  for (const key of Object.keys(itemExportMap)) {
    const sensorId = Number(key)
    urls[sensorId] = itemExportMap[sensorId].url as string
    const labelsExport = itemExportMap[sensorId].labels
    if (labelsExport !== undefined) {
      for (const labelExport of labelsExport) {
        const labelId = labelExport.id.toString()
        // Each label may appear in multiple sensors
        if (tracking && labelExport.id in labels) {
          labels[labelId].sensors.push(sensorId)
          continue
        }

        const categories: number[] = []
        if (labelExport.category in categoryNameMap) {
          categories.push(categoryNameMap[labelExport.category])
        }

        const attributes = parseExportAttributes(
          labelExport.attributes,
          attributeNameMap,
          attributeValueMap
        )

        const [importedLabel, importedShapes] = convertLabelToImport(
          labelExport,
          itemIndex,
          sensorId,
          categories,
          attributes
        )

        if (tracking) {
          importedLabel.track = labelExport.id.toString()
        }

        labels[labelId] = importedLabel
        for (const indexedShape of importedShapes) {
          shapes[indexedShape.id] = indexedShape
        }
      }
    }
  }

  return makeItem(
    {
      urls,
      index: itemIndex,
      id: itemId.toString(),
      timestamp,
      videoName,
      labels,
      shapes
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
  attributesExport: { [key: string]: string[] | boolean },
  attributeNameMap: { [key: string]: [number, Attribute] },
  attributeValueMap: { [key: string]: number }
): { [key: number]: number[] } {
  const labelAttributes: { [key: number]: number[] } = {}
  Object.entries(attributesExport).forEach(([name, attributeList]) => {
    // Get the config attribute that matches the exported attribute name
    if (name in attributeNameMap) {
      const [configIndex, currentAttribute] = attributeNameMap[name]
      // Load the attribute based on its type
      if (currentAttribute.toolType === AttributeToolType.SWITCH) {
        // Boolean attribute case- only two choices, not a list
        let value = 0
        const attributeVal = attributeList as boolean
        if (attributeVal) {
          value = 1
        }
        labelAttributes[configIndex] = [value]
      } else if (
        currentAttribute.toolType === AttributeToolType.LIST ||
        currentAttribute.toolType === AttributeToolType.LONG_LIST
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
 */
function convertLabelToImport(
  labelExport: LabelExport,
  item: number,
  sensorId: number,
  category?: number[],
  attributes?: { [key: number]: number[] }
): [LabelType, ShapeType[]] {
  let labelType = LabelTypeName.EMPTY
  let shapes: null | ShapeType[] = null
  const labelId = labelExport.id.toString()

  /**
   * Convert each import shape based on their type
   * TODO: no polyline2d
   */
  if (labelExport.box2d !== null) {
    labelType = LabelTypeName.BOX_2D
    shapes = [makeRect(labelExport.box2d)]
  } else if (labelExport.poly2d !== null) {
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
  } else if (labelExport.box3d !== null) {
    labelType = LabelTypeName.BOX_3D
    shapes = [makeCube(labelExport.box3d)]
  } else if (labelExport.plane3d !== null) {
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
      manual: labelExport.manualShape,
      category,
      attributes,
      sensors: [sensorId]
    },
    false
  )

  return [labelImport, shapeImports]
}

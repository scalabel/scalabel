import _ from 'lodash'
import { AttributeToolType, LabelTypeName } from '../common/types'
import { PointType } from '../drawable/2d/path_point2d'
import { makeCube, makeItem, makeLabel, makePlane, makePolygon, makePolyPathPoint, makeRect } from '../functional/states'
import { Attribute, IdType, ItemType, LabelIdMap, LabelType, ShapeIdMap, ShapeType } from '../functional/types'
import { ItemExport, LabelExport } from './bdd_types'

/**
 * Converts single exported item to frontend state format
 * @param item the item in export format
 * @param itemIndex the item index (relative to task)
 * @param itemId the item id (relative to project)
 * @param attributesNameMap look up an attribute and its index from its name
 * @param attributeValueMap look up an attribute value's index
 * @param categoryNameMap look up a category's index from its name
 */
export function convertItemToImport (
  videoName: string,
  timestamp: number,
  itemExportMap: {[id: number]: Partial<ItemExport>},
  itemIndex: number, itemId: number,
  attributeNameMap: {[key: string]: [number, Attribute]},
  attributeValueMap: {[key: string]: number},
  categoryNameMap: {[key: string]: number},
  tracking: boolean
): ItemType {
  const urls: {[id: number]: string} = {}

  const labels: LabelIdMap = {}
  const shapes: ShapeIdMap = {}
  for (const key of Object.keys(itemExportMap)) {
    const sensorId = Number(key)
    urls[sensorId] = itemExportMap[sensorId].url as string
    const labelsExport = itemExportMap[sensorId].labels
    if (labelsExport) {
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

        const [importedLabel, importedShapes] =
          convertLabelToImport(
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

  return makeItem({
    urls,
    index: itemIndex,
    id: itemId.toString(),
    timestamp,
    videoName,
    labels,
    shapes
  }, true)
}

/**
 * parses attributes from BDD format (strings)
 * to internal format (index in config's attributes)
 * @param attributesExport the attributes to process
 * @param attributesNameMap look up an attribute and its index from its name
 * @param attributeValueMap look up an attribute value's index
 */
function parseExportAttributes (
  attributesExport: {[key: string]: (string[] | boolean) },
  attributeNameMap: {[key: string]: [number, Attribute]},
  attributeValueMap: {[key: string]: number}):
  {[key: number]: number[] } {
  const labelAttributes: {[key: number]: number[]} = {}
  Object.entries(attributesExport).forEach(([name, attributeList]) => {
    // get the config attribute that matches the exported attribute name
    if (name in attributeNameMap) {
      const [configIndex, currentAttribute] = attributeNameMap[name]
      // load the attribute based on its type
      if (currentAttribute.toolType === AttributeToolType.SWITCH) {
        // boolean attribute case- only two choices, not a list
        let value = 0
        const attributeVal = attributeList as boolean
        if (attributeVal === true) {
          value = 1
        }
        labelAttributes[configIndex] = [value]
      } else if (currentAttribute.toolType === AttributeToolType.LIST
        || currentAttribute.toolType === AttributeToolType.LONG_LIST) {
        // list attribute case- can choose multiple values
        const selectedIndices: number[] = []
        const attributeValues = attributeList as string[]
        attributeValues.forEach((value: string) => {
          // get the index of the selected value
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
  * @param label the label in export format
  * @param shapesImport map to update, from shapeId to shape
  */
function convertLabelToImport (
  labelExport: LabelExport,
  item: number,
  sensorId: number,
  category?: number[],
  attributes?: {[key: number]: number[]}
): [LabelType, ShapeType[]] {
  let labelType = LabelTypeName.EMPTY
  let shapeData: null | ShapeType = null
  const labelId = labelExport.id.toString()

  /**
   * Convert each import shape based on their type
   * TODO: no polyline2d
   */
  if (labelExport.box2d) {
    labelType = LabelTypeName.BOX_2D
    shapeData = makeRect(labelExport.box2d)
  } else if (labelExport.poly2d) {
    const polyExport = labelExport.poly2d[0]
    labelType = (polyExport.closed) ?
      LabelTypeName.POLYGON_2D : LabelTypeName.POLYLINE_2D
    const points = polyExport.vertices.map(
      (vertex, i) => makePolyPathPoint({
        x: vertex[0],
        y: vertex[1],
        pointType: (polyExport.types[i] === 'L') ?
          PointType.VERTEX : PointType.CURVE
      })
    )
    shapeData = makePolygon({ points })
  } else if (labelExport.box3d) {
    labelType = LabelTypeName.BOX_3D
    shapeData = makeCube(labelExport.box3d)
  } else if (labelExport.plane3d) {
    labelType = LabelTypeName.PLANE_3D
    shapeData = makePlane(labelExport.plane3d)
  }

  // if the label has any shapes, import them too
  const shapeIds: IdType[] = []
  const shapeImports: ShapeType[] = []
  if (shapeData !== null) {
    shapeData.label.push(labelId)
    shapeIds.push(shapeData.id)
    shapeImports.push(shapeData)
  }

  const labelImport = makeLabel({
    id: labelId,
    type: labelType,
    item,
    shapes: shapeIds,
    manual: labelExport.manualShape,
    category,
    attributes,
    sensors: [sensorId]
  }, true)

  return [labelImport, shapeImports]
}

import _ from 'lodash'
import { AttributeToolType, LabelTypeName, ShapeTypeName } from '../common/types'
import { PointType } from '../drawable/2d/path_point2d'
import { ItemExport, LabelExport } from '../functional/bdd_types'
import { makeItem, makeLabel, makePathPoint } from '../functional/states'
import { Attribute, IndexedShapeType,
  ItemType, LabelType } from '../functional/types'

/**
 * Converts single exported item to frontend state format
 * @param item the item in export format
 * @param itemInd the item index (relative to task)
 * @param itemId the item id (relative to project)
 * @param attributesNameMap look up an attribute and its index from its name
 * @param attributeValueMap look up an attribute value's index
 * @param categoryNameMap look up a category's index from its name
 */
export function convertItemToImport (
  videoName: string,
  timestamp: number,
  itemExportMap: {[id: number]: Partial<ItemExport>},
  itemInd: number, itemId: number,
  attributeNameMap: {[key: string]: [number, Attribute]},
  attributeValueMap: {[key: string]: number},
  categoryNameMap: {[key: string]: number},
  maxLabelId: number,
  maxShapeId: number,
  tracking: boolean
): [ItemType, number, number] {
  const urls: {[id: number]: string} = {}

  const labelExportIdToImportId: { [key: number]: number} = {}
  const labelImports: { [key: number]: LabelType } = {}
  const shapeImports: { [key: number]: IndexedShapeType } = {}
  for (const key of Object.keys(itemExportMap)) {
    const sensorId = Number(key)
    urls[sensorId] = itemExportMap[sensorId].url as string
    const labelsExport = itemExportMap[sensorId].labels
    if (labelsExport) {
      for (const labelExport of labelsExport) {
        if (tracking && labelExport.id in labelExportIdToImportId) {
          const labelId = labelExportIdToImportId[labelExport.id]
          labelImports[labelId].sensors.push(sensorId)
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

        const [labelImport, indexedShapeImports, newMaxShapeId] =
          convertLabelToImport(
            labelExport,
            maxLabelId + 1,
            itemInd,
            maxShapeId,
            sensorId,
            categories,
            attributes
          )

        if (tracking) {
          labelImport.track = labelExport.id
        }

        labelExportIdToImportId[labelExport.id] = labelImport.id
        labelImports[maxLabelId + 1] = labelImport
        for (const indexedShape of indexedShapeImports) {
          shapeImports[indexedShape.id] = indexedShape
        }

        maxShapeId = newMaxShapeId
        maxLabelId++
      }
    }
  }

  const partialItemImport: Partial<ItemType> = {
    urls,
    index: itemInd,
    id: itemId,
    timestamp
  }
  partialItemImport.videoName = videoName
  const itemImport = makeItem(partialItemImport)

  itemImport.labels = labelImports
  itemImport.shapes = shapeImports

  return [itemImport, maxLabelId, maxShapeId]
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
  * based on the label in export format, create a label in internal formt
  * and update the corresponding shapes in the map
  * @param label the label in export format
  * @param shapesImport map to update, from shapeId to shape
  */
function convertLabelToImport (
  labelExport: LabelExport,
  labelId: number,
  item: number,
  maxShapeId: number,
  sensorId: number,
  category?: number[],
  attributes?: {[key: number]: number[]}
): [LabelType, IndexedShapeType[], number] {
  let shapeType = ShapeTypeName.UNKNOWN
  let labelType = LabelTypeName.EMPTY

  const shapeImports: IndexedShapeType[] = []
  const shapeIds: number[] = []
  if (labelExport.box2d) {
    shapeType = ShapeTypeName.RECT
    labelType = LabelTypeName.BOX_2D
    shapeImports.push({
      id: maxShapeId + 1,
      label: [labelId],
      type: shapeType,
      shape: labelExport.box2d
    })
    maxShapeId++
    shapeIds.push(shapeImports[0].id)
  } else if (labelExport.poly2d) {
    shapeType = ShapeTypeName.POLYGON_2D
    const polyExport = labelExport.poly2d[0]
    labelType = (polyExport.closed) ?
      LabelTypeName.POLYGON_2D : LabelTypeName.POLYLINE_2D
    const points = polyExport.vertices.map(
      (vertex, i) => makePathPoint({
        x: vertex[0],
        y: vertex[1],
        type: (polyExport.types[i] === 'L') ?
          PointType.VERTEX : PointType.CURVE
      })
    )
    for (const point of points) {
      shapeImports.push({
        id: maxShapeId + 1,
        label: [labelId],
        type: shapeType,
        shape: point
      })
      maxShapeId++
      shapeIds.push(maxShapeId)
    }
  } else if (labelExport.box3d) {
    shapeType = ShapeTypeName.CUBE
    labelType = LabelTypeName.BOX_3D
    shapeImports.push({
      id: maxShapeId + 1,
      label: [labelId],
      type: shapeType,
      shape: labelExport.box3d
    })
    maxShapeId++
    shapeIds.push(shapeImports[0].id)
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
  })

  return [labelImport, shapeImports, maxShapeId]
}

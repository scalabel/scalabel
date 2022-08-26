import { LabelTypeName, ShapeTypeName } from "../../src/const/common"
import { makePathPoint2D } from "../../src/functional/states"
import {
  box3dToCube,
  transformBox3D
} from "../../src/server/bdd_type_transformers"
import {
  convertItemToExport,
  convertPolygonToExport,
  convertStateToExport
} from "../../src/server/export"
import { PathPointType } from "../../src/types/state"
import {
  sampleItemExportImageTagging,
  sampleItemExportImage,
  sampleItemExportImagePolygon,
  sampleStateExportImage,
  sampleStateExportImagePolygon,
  sampleItemExportImage3dBox,
  sampleStateExportImage3dBox,
  sampleStateExportImagePolygonMulti
} from "../test_states/test_export_objects"
import { readSampleState } from "./util/io"

const sampleStateFile = "./app/test/test_states/sample_state.json"
const samplePolygonStateFile =
  "./app/test/test_states/sample_state_polygon.json"
const samplePolygonMultiStateFile =
  "./app/test/test_states/sample_state_polygon_multi_track.json"
const sampleTagStateFile = "./app/test/test_states/sample_state_tag.json"
const sample3dBoxStateFile = "./app/test/test_states/sample_state_3d_box.json"

describe("test export functionality across multiple labeling types", () => {
  test("unit test for polygon export", () => {
    const points = [
      makePathPoint2D({ x: 0, y: 1, pointType: PathPointType.LINE }),
      makePathPoint2D({ x: 0, y: 2, pointType: PathPointType.CURVE })
    ]
    const labelType = LabelTypeName.POLYGON_2D
    const polyExport = convertPolygonToExport(points, labelType)
    const polyPoint = polyExport[0]
    expect(polyPoint.closed).toBe(true)
    expect(polyPoint.types).toBe("LC")
    expect(polyPoint.vertices).toEqual([
      [0, 1],
      [0, 2]
    ])
  })
})

describe("test export functionality for polygon", () => {
  test("export multi-component polygons in track mode", () => {
    const state = readSampleState(samplePolygonMultiStateFile)
    const exportedState = convertStateToExport(state)
    expect(exportedState).toEqual(sampleStateExportImagePolygonMulti)
  })
})

describe("test export functionality for image tagging", () => {
  test("single item conversion", () => {
    const state = readSampleState(sampleTagStateFile)
    const config = state.task.config
    const item = state.task.items[0]
    const itemExport = convertItemToExport(config, item)[0]
    expect(itemExport).toEqual(sampleItemExportImageTagging)
  })
})

describe("test export functionality for bounding box", () => {
  test("single item conversion", () => {
    const state = readSampleState(sampleStateFile)
    const config = state.task.config
    const item = state.task.items[0]
    const itemExport = convertItemToExport(config, item)[0]
    expect(itemExport).toEqual(sampleItemExportImage)
  })
  test("full state export with empty items", () => {
    const state = readSampleState(sampleStateFile)
    const exportedState = convertStateToExport(state)
    expect(exportedState).toEqual(sampleStateExportImage)
  })
})

describe("test export functionality for segmentation", () => {
  test("single item conversion", () => {
    const state = readSampleState(samplePolygonStateFile)
    const config = state.task.config
    const item = state.task.items[0]
    const itemExport = convertItemToExport(config, item)[0]
    expect(itemExport).toEqual(sampleItemExportImagePolygon)
  })
  test("full state export with empty items", () => {
    const state = readSampleState(samplePolygonStateFile)
    const exportedState = convertStateToExport(state)
    expect(exportedState).toEqual(sampleStateExportImagePolygon)
  })
})

describe("test export functionality for tracking", () => {
  test("single item conversion", () => {})
  test("full state export including empty items", () => {})
})

describe("test export functionality for 3d bounding box", () => {
  test("single item conversion", () => {
    const state = readSampleState(sample3dBoxStateFile)
    const config = state.task.config
    const item = state.task.items[0]
    const itemExport = convertItemToExport(config, item)[0]
    expect(itemExport).toEqual(sampleItemExportImage3dBox)
  })
  test("full state export including empty items", () => {
    const state = readSampleState(sample3dBoxStateFile)
    const exportedState = convertStateToExport(state)
    expect(exportedState).toEqual(sampleStateExportImage3dBox)
  })
  test("item import and export", () => {
    const labels = sampleStateExportImage3dBox[0].labels
    const boxes = labels.map((l) => l.box3d)
    for (const box of boxes) {
      expect(box).not.toBeNull()
      if (box !== null) {
        const importedBox = box3dToCube(box)
        const importedShape = {
          id: "0",
          label: [] as string[],
          shapeType: ShapeTypeName.CUBE,
          ...importedBox
        }
        const exportedBox = transformBox3D(importedShape)
        expect(box.dimension).toEqual(exportedBox.dimension)
        expect(box.orientation).toEqual(exportedBox.orientation)
        expect(box.location).toEqual(exportedBox.location)
      }
    }
  })
})

import * as fs from 'fs-extra'
import _ from 'lodash'
import { LabelTypeName } from '../../js/const/common'
import { makePathPoint2D } from '../../js/functional/states'
import {
  convertItemToExport,
  convertPolygonToExport, convertStateToExport
} from '../../js/server/export'
import { PathPointType, State } from '../../js/types/state'
import {
  sampleItemExportImage, sampleItemExportImagePolygon,
  sampleStateExportImage, sampleStateExportImagePolygon
} from '../test_states/test_export_objects'

const sampleStateFile = './app/src/test/test_states/sample_state.json'
const samplePolygonStateFile = './app/src/test/test_states/sample_state_polygon.json'

describe('test export functionality across multiple labeling types', () => {
  test('unit test for polygon export', () => {
    const points = [
      makePathPoint2D({ x: 0, y: 1, pointType: PathPointType.LINE }),
      makePathPoint2D({ x: 0, y: 2, pointType: PathPointType.CURVE })
    ]
    const labelType = LabelTypeName.POLYGON_2D
    const polyExport = convertPolygonToExport(points, labelType)
    const polyPoint = polyExport[0]
    expect(polyPoint.closed).toBe(true)
    expect(polyPoint.types).toBe('LC')
    expect(polyPoint.vertices).toEqual([[0, 1], [0, 2]])
  })
  describe('test export functionality for bounding box', () => {
    test('single item conversion', () => {
      const state = readSampleState(sampleStateFile)
      const config = state.task.config
      const item = state.task.items[0]
      const itemExport = convertItemToExport(config, item)[0]
      expect(itemExport).toEqual(sampleItemExportImage)
    }),
      test('full state export with empty items', () => {
        const state = readSampleState(sampleStateFile)
        const exportedState = convertStateToExport(state)
        expect(exportedState).toEqual(sampleStateExportImage)
      })
  }),
    describe('test export functionality for segmentation', () => {
      test('single item conversion', () => {
        const state = readSampleState(samplePolygonStateFile)
        const config = state.task.config
        const item = state.task.items[0]
        const itemExport = convertItemToExport(config, item)[0]
        expect(itemExport).toEqual(sampleItemExportImagePolygon)
      }),
        test('full state export with empty items', () => {
          const state = readSampleState(samplePolygonStateFile)
          const exportedState = convertStateToExport(state)
          expect(exportedState).toEqual(sampleStateExportImagePolygon)
        })
    }),
    describe('test export functionality for tracking', () => {
      test('single item conversion', () => {
        return
      }),
        test('full state export including empty items', () => {
          return
        })
    })
})

/**
 * helper function to read sample state
 */
function readSampleState (fileName: string): State {
  return JSON.parse(fs.readFileSync(fileName, 'utf8'))
}

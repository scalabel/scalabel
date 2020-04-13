import * as fs from 'fs-extra'
import _ from 'lodash'
import Session from '../../js/common/session'
import { LabelTypeName } from '../../js/common/types'
import { PathPoint2D, PointType } from '../../js/drawable/2d/path_point2d'
import { makePolygon } from '../../js/functional/states'
import { State } from '../../js/functional/types'
import { convertItemToExport,
  convertPolygonToExport, convertStateToExport } from '../../js/server/export'
import { sampleItemExportImage, sampleItemExportImagePolygon,
         sampleStateExportImage, sampleStateExportImagePolygon} from '../test_export_objects'

beforeAll(() => {
  Session.devMode = false
})
const sampleStateFile = './app/src/test/sample_state.json'
const samplePolygonStateFile = './app/src/test/sample_state_polygon.json'

describe('test export functionality across multiple labeling types', () => {
  test('unit test for polygon export', () => {
    const points = [
      (new PathPoint2D(0, 1, PointType.VERTEX)).toPathPoint(),
      (new PathPoint2D(0, 2, PointType.CURVE)).toPathPoint()
    ]
    const poly2d = makePolygon({ points })
    const labelType = LabelTypeName.POLYGON_2D
    const polyExport = convertPolygonToExport(poly2d, labelType)
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

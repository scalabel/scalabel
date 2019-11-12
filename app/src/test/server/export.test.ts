import * as fs from 'fs-extra'
import _ from 'lodash'
import Session from '../../js/common/session'
import { State } from '../../js/functional/types'
import { convertItemToExport, convertStateToExport } from '../../js/server/export'
import { sampleItemExportImage, sampleStateExportImage } from '../test_export_objects'

beforeAll(() => {
  Session.devMode = false
})

describe('test export functionality across multiple labeling types', () => {
  describe('test export functionality for bounding box', () => {
    test('single item conversion', () => {
      const state = readSampleState()
      const config = state.task.config
      const item = state.task.items[0]
      const itemExport = convertItemToExport(config, item)[0]
      expect(itemExport).toEqual(sampleItemExportImage)
    }),
      test('full state export with empty items', () => {
        const state = readSampleState()
        const exportedState = convertStateToExport(state)
        expect(exportedState).toEqual(sampleStateExportImage)
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
function readSampleState (): State {
  return JSON.parse(fs.readFileSync('./app/src/test/sample_state.json', 'utf8'))
}

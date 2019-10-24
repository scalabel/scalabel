import * as fs from 'fs-extra'
import _ from 'lodash'
import Session from '../../js/common/session'
import { State } from '../../js/functional/types'
import { convertItemToExport, convertStateToExport } from '../../js/server/export'
import { sampleItemExport, sampleStateExport } from '../test_export_objects'

beforeAll(() => {
  Session.devMode = false
})

describe('test export functionality for bounding box', () => {
  describe('single item conversion to export', () => {
    test('single item conversion', () => {
      const state = readSampleState()
      const config = state.task.config
      const item = state.task.items[0]
      const itemExport = convertItemToExport(config, item)
      expect(itemExport).toEqual(sampleItemExport)
    })

    test('single item conversion with altered export', () => {
      const oldExportName = sampleItemExport.name
      const oldExportUrl = sampleItemExport.url
      const state = readSampleState()
      const config = state.task.config
      const item = state.task.items[0]
      item.url = 'altered'
      sampleItemExport.name = 'altered'
      sampleItemExport.url = 'altered'
      const itemExport = convertItemToExport(config, item)
      expect(itemExport).toEqual(sampleItemExport)
      sampleItemExport.name = oldExportName
      sampleItemExport.url = oldExportUrl
    })
  }),
  test('full state export with empty items', () => {
    const state = readSampleState()
    const exportedState = convertStateToExport(state)
    expect(exportedState).toEqual(sampleStateExport)
  })
})

/**
 * helper function to read sample state
 */
function readSampleState (): State {
  return JSON.parse(fs.readFileSync('./app/src/test/sample_state.json', 'utf8'))
}

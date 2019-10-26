import { withStyles } from '@material-ui/core'
import * as fs from 'fs-extra'
import { ChangeEvent } from 'react'
import Session, { ConnectionStatus } from '../../js/common/session'
import { initFromJson } from '../../js/common/session_init'
import { Synchronizer } from '../../js/common/synchronizer'
import CreateForm from '../../js/components/create_form'
import { ItemExport } from '../../js/functional/bdd_types'
import { State } from '../../js/functional/types'
import { Endpoint, FormField } from '../../js/server/types'
import { formStyle } from '../../js/styles/create'

export interface TestConfig {
  /** project name for current test */
  projectName: string
  /** task index for current test */
  taskIndex: number
  /** path to examples folder */
  examplePath: string
  /** path to test directory */
  testDirPath: string
  /** example export path */
  sampleExportPath: string
  /** example export filename */
  sampleExportFilename: string
  /** item list filename */
  itemListFilename: string
  /** categories filename */
  categoriesFilename: string
  /** attributes filename */
  attributesFilename: string
  /** if we are exporting */
  exportMode: boolean
}

export let testConfig: TestConfig = {
  projectName: 'integration-test',
  taskIndex: 0,
  examplePath: './examples/',
  testDirPath: './test_data/',
  sampleExportPath: './app/src/test/',
  sampleExportFilename: 'sample_export.json',
  itemListFilename: 'image_list.yml',
  categoriesFilename: 'categories.yml',
  attributesFilename: 'bbox_attributes.yml',
  exportMode: false
}

/**
 * changes current test config to newConfig
 * @param newConfig
 */
export function changeTestConfig (newConfig: Partial<TestConfig>) {
  testConfig = {
    ...testConfig,
    ...newConfig
  }
}

/**
 * helper function to get example file from disc
 * @param filename
 */
function getExampleFileFromDisc (
  filename: string,
  examplePath = testConfig.examplePath
): File {
  const fileAsString = fs.readFileSync(examplePath + filename, {
    encoding: 'utf8'
  })
  return new File([fileAsString], filename)
}

/**
 * gets true export data from disc
 */
export function getExportFromDisc () {
  return JSON.parse(
    fs.readFileSync(
      testConfig.sampleExportPath + testConfig.sampleExportFilename,
      'utf8'
    )
  )
}

/**
 * deep deletes timestamp from given data
 * @param data
 */
// tslint:disable-next-line: no-any
export function deepDeleteTimestamp (data: ItemExport[]): any[] {
  const copy = JSON.parse(JSON.stringify(data))
  for (const entry of copy) {
    if (entry) {
      delete entry.timestamp
    }
  }
  return copy
}

/**
 * helper function to force javascript to sleep
 * @param milliseconds
 */
export function sleep (milliseconds: number): Promise<object> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds))
}

/**
 * helper function to wait for backend to save
 */
export function waitForSave (): Promise<object> {
  return new Promise(async (resolve) => {
    while (Session.status !== ConnectionStatus.SAVED) {
      await sleep(10)
    }
    resolve()
  })
}

/**
 * Deletes the generetaed test directory
 */
export function deleteTestDir (): void {
  fs.removeSync(testConfig.testDirPath)
  return
}

/**
 * init session for integration test
 */
export async function projectInitSession (): Promise<Synchronizer> {
  return new Promise<Synchronizer>((resolve) => {
    const synchronizer = new Synchronizer(
      testConfig.taskIndex,
      testConfig.projectName,
      (state: State) => {
        initFromJson(state, synchronizer.middleware)
        resolve(synchronizer)
      }
    )
  })
}

/**
 * gets exported annotations as a string
 */
export async function getExport (): Promise<ItemExport[]> {
  return new Promise<ItemExport[]>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.onreadystatechange = () => {
      if (xhr.readyState === 4) {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.response))
        } else {
          reject({
            status: xhr.status,
            statusText: xhr.statusText
          })
        }
      } else {
        return
      }
    }
    xhr.open('GET', Endpoint.EXPORT + '?' +
    FormField.PROJECT_NAME + '=' + testConfig.projectName)
    xhr.send()
  })
}

/**
 * over writes CreateForm
 */
class IntegrationCreateForm extends CreateForm {
  /**
   * over writes create form get form data to input files into ajax request
   * @param event
   */
  protected getFormData (event: ChangeEvent<HTMLFormElement>): FormData {
    let itemFilePath: string
    let itemFilename: string
    if (testConfig.exportMode) {
      itemFilePath = testConfig.sampleExportPath
      itemFilename = testConfig.sampleExportFilename
    } else {
      itemFilePath = testConfig.examplePath
      itemFilename = testConfig.itemListFilename
    }
    const itemFile = getExampleFileFromDisc(itemFilename, itemFilePath)
    const categoriesFile = getExampleFileFromDisc(
      testConfig.categoriesFilename
    )
    const attributesFile = getExampleFileFromDisc(
      testConfig.attributesFilename
    )
    const formData = new FormData(event.target)
    formData.delete('item_file')
    formData.append('item_file', itemFile, itemFile.name)
    formData.delete('categories')
    formData.append('categories', categoriesFile, categoriesFile.name)
    formData.delete('attributes')
    formData.append('attributes', attributesFile, attributesFile.name)
    return formData
  }
}

export const StyledIntegrationForm = withStyles(formStyle)(
  IntegrationCreateForm
)

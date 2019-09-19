import { withStyles } from '@material-ui/core'
import * as fs from 'fs-extra'
import { ChangeEvent } from 'react'
import Session, { ConnectionStatus } from '../../js/common/session'
import CreateForm from '../../js/components/create_form'
import { DashboardContents } from '../../js/components/dashboard'
import { formStyle } from '../../js/styles/create'

// TODO: Remove no any's when exporting is implemented with node

export interface TestConfig {
  /** project name for current test */
  projectName: string
  /** task index for current test */
  taskIndex: number
  /** path to examples folder */
  examplePath: string
  /** export url */
  exportUrl: string
  /** path to test directory */
  testDirPath: string
  /** item list filename */
  itemListFilename: string
  /** categories filename */
  categoriesFilename: string
  /** attributes filename */
  attributesFilename: string
}

export let testConfig: TestConfig = {
  projectName: 'integration-test',
  taskIndex: 0,
  examplePath: './examples/',
  exportUrl: './postExportV2?project_name=integration-test',
  testDirPath: './test_data/',
  itemListFilename: 'image_list.yml',
  categoriesFilename: 'categories.yml',
  attributesFilename: 'bbox_attributes.yml'
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
function getExampleFileFromDisc (filename: string): File {
  const examplePath = testConfig.examplePath
  const fileAsString = fs.readFileSync(examplePath + filename, {
    encoding: 'utf8'
  })
  return new File([fileAsString], filename)
}

/**
 * gets true export data from disc
 */
export function getExportFromDisc () {
  const sampleExportPath = './app/src/test/sample_export.json'
  return JSON.parse(fs.readFileSync(sampleExportPath, 'utf8'))
}

/**
 * deep deletes timestamp from given data
 * @param data
 */
// tslint:disable-next-line: no-any
export function deepDeleteTimestamp (data: any[]) {
  for (const entry of data) {
    if (entry) {
      delete entry.timestamp
    }
  }
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
export async function projectInitSession (): Promise<{}> {
  return new Promise<{}>((resolve, reject) => {
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

    // send the request to the back end
    const request = JSON.stringify({
      task: {
        index: testConfig.taskIndex,
        projectOptions: { name: testConfig.projectName }
      }
    })
    xhr.open('POST', './postLoadAssignmentV2', true)
    xhr.send(request)
  })
}

/**
 * init dashboard for integration test
 */
export async function projectInitDashboard (): Promise<DashboardContents> {
  return new Promise<DashboardContents>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.onreadystatechange = () => {
      if (xhr.readyState === 4) {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.response) as DashboardContents)
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

    // send the request to the back end
    const request = JSON.stringify({
      name: testConfig.projectName
    })
    xhr.open('POST', './postDashboardContents', true)
    xhr.send(request)
  })
}

/**
 * gets exported annotations as a string
 */
// tslint:disable-next-line: no-any
export async function getExport (): Promise<any[]> {
  // tslint:disable-next-line: no-any
  return new Promise<any[]>((resolve, reject) => {
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
    xhr.open('GET', testConfig.exportUrl)
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
    const itemFile = getExampleFileFromDisc(testConfig.itemListFilename)
    const categoriesFile = getExampleFileFromDisc(testConfig.categoriesFilename)
    const attributesFile = getExampleFileFromDisc(testConfig.attributesFilename)
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

import { withStyles } from "@material-ui/core"
import { readdir } from "fs"
import * as fs from "fs-extra"
import * as path from "path"
import { ChangeEvent } from "react"
import * as util from "util"

import Session from "../../src/common/session"
import CreateForm from "../../src/components/create_form"
import { Endpoint } from "../../src/const/connection"
import { FormField } from "../../src/const/project"
import { isStatusSaved } from "../../src/functional/selector"
import { getTaskDir } from "../../src/server/path"
import { formStyle } from "../../src/styles/create"
import { ItemExport } from "../../src/types/export"

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
  samplePath: string
  /** example export filename */
  sampleExportFilename: string
  /** example project json filename */
  sampleProjectJsonFilename: string
  /** item list filename */
  itemListFilename: string
  /** categories filename */
  categoriesFilename: string
  /** attributes filename */
  attributesFilename: string
  /** filename for storing project info */
  projectFilename: string
  /** if we are exporting */
  exportMode: boolean
}

export let testConfig: TestConfig = {
  projectName: "integration-test",
  taskIndex: 0,
  examplePath: "./examples/",
  testDirPath: "./test_data/",
  samplePath: "./app/test/test_states",
  sampleExportFilename: "sample_export.json",
  sampleProjectJsonFilename: "sample_project.json",
  itemListFilename: "image_list.yml",
  categoriesFilename: "categories.yml",
  attributesFilename: "bbox_attributes.yml",
  projectFilename: "project.json",
  exportMode: false
}

/**
 * changes current test config to newConfig
 *
 * @param newConfig
 */
export function changeTestConfig(newConfig: Partial<TestConfig>): void {
  testConfig = {
    ...testConfig,
    ...newConfig
  }
}

/**
 * helper function to get example file from disc
 *
 * @param filename
 * @param examplePath
 */
function getExampleFileFromDisc(
  filename: string,
  examplePath = testConfig.examplePath
): File {
  const fileAsString = fs.readFileSync(examplePath + filename, {
    encoding: "utf8"
  })
  return new File([fileAsString], filename)
}

/**
 * gets true export data from disc
 */
export function getExportFromDisc(): ItemExport[] {
  return JSON.parse(
    fs.readFileSync(
      path.join(testConfig.samplePath, testConfig.sampleExportFilename),
      "utf8"
    )
  ) as ItemExport[]
}

/**
 * gets true project data from disc
 */
export function getProjectJsonFromDisc(): void {
  return JSON.parse(
    fs.readFileSync(
      path.join(testConfig.samplePath, testConfig.sampleProjectJsonFilename),
      "utf-8"
    )
  )
}

/**
 * deep deletes timestamp from given data
 *
 * @param data
 */
export function deepDeleteTimestamp(data: ItemExport[]): unknown[] {
  const copy = JSON.parse(JSON.stringify(data))
  for (const entry of copy) {
    if (entry !== undefined) {
      delete entry.timestamp
    }
  }
  return copy
}

/**
 * helper function to force javascript to sleep
 *
 * @param milliseconds
 */
export async function sleep(milliseconds: number): Promise<void> {
  return await new Promise((resolve) => setTimeout(resolve, milliseconds))
}

/**
 * helper function to wait for backend to save
 */
export async function waitForSave(): Promise<void> {
  while (!isStatusSaved(Session.store.getState())) {
    await sleep(10)
  }
}

/**
 * Deletes the generetaed test directory
 */
export function deleteTestDir(): void {
  fs.removeSync(testConfig.testDirPath)
}

/**
 * gets exported annotations as a string
 */
export async function getExport(): Promise<ItemExport[]> {
  return await new Promise<ItemExport[]>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.onreadystatechange = () => {
      if (xhr.readyState === 4) {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.response))
        } else {
          reject(new Error(xhr.statusText))
        }
      } else {
      }
    }
    xhr.open(
      "GET",
      Endpoint.EXPORT +
        "?" +
        FormField.PROJECT_NAME +
        "=" +
        testConfig.projectName
    )
    xhr.send()
  })
}

/**
 * gets created project.json file
 */
export function getProjectJson(): unknown {
  return JSON.parse(
    fs.readFileSync(
      path.join(
        testConfig.testDirPath,
        testConfig.projectName,
        testConfig.projectFilename
      ),
      "utf-8"
    )
  )
}

/**
 * Counts created task/{i}.json file
 *
 * @param projectName
 */
export async function countTasks(projectName: string): Promise<number> {
  const taskDir = path.join(testConfig.testDirPath, getTaskDir(projectName))
  if (!(await fs.pathExists(taskDir))) {
    return 0
  }

  const readdirPromise = util.promisify(readdir)
  const dirEnts = await readdirPromise(taskDir)
  return dirEnts.length
}

/**
 * over writes CreateForm
 */
class IntegrationCreateForm extends CreateForm {
  /**
   * over writes create form get form data to input files into ajax request
   *
   * @param event
   */
  protected getFormData(event: ChangeEvent<HTMLFormElement>): FormData {
    let itemFilePath: string
    let itemFilename: string
    if (testConfig.exportMode) {
      itemFilePath = testConfig.samplePath
      itemFilename = testConfig.sampleExportFilename
    } else {
      itemFilePath = testConfig.examplePath
      itemFilename = testConfig.itemListFilename
    }
    const itemFile = getExampleFileFromDisc(itemFilename, itemFilePath)
    const categoriesFile = getExampleFileFromDisc(testConfig.categoriesFilename)
    const attributesFile = getExampleFileFromDisc(testConfig.attributesFilename)
    const formData = new FormData(event.target)
    formData.delete("item_file")
    formData.append("item_file", itemFile, itemFile.name)
    formData.delete("sensors_file")
    formData.delete("categories")
    formData.append("categories", categoriesFile, categoriesFile.name)
    formData.delete("attributes")
    formData.append("attributes", attributesFile, attributesFile.name)

    return formData
  }
}

export const StyledIntegrationForm = withStyles(formStyle)(
  IntegrationCreateForm
)

import * as fs from "fs-extra"
import * as path from "path"

import { addBox2dLabel } from "../../../src/action/box2d"
import { StorageStructure } from "../../../src/const/storage"
import {
  makeItem,
  makeSensor,
  makeState,
  makeTask
} from "../../../src/functional/states"
import { parseConfig } from "../../../src/server/config"
import { AddLabelsAction } from "../../../src/types/action"
import { ServerConfig } from "../../../src/types/config"
import {
  RectType,
  State,
  TaskType,
  Vector3Type
} from "../../../src/types/state"

/**
 * Check equality between two Vector3Type objects
 *
 * @param v1
 * @param v2
 * @param num
 */
export function expectVector3TypesClose(
  v1: Vector3Type,
  v2: Vector3Type,
  num = 2
): void {
  expect(v1.x).toBeCloseTo(v2.x, num)
  expect(v1.y).toBeCloseTo(v2.y, num)
  expect(v1.z).toBeCloseTo(v2.z, num)
}

/**
 * Check that rectangles are close
 *
 * @param r1
 * @param r2
 * @param num
 */
export function expectRectTypesClose(
  r1: RectType,
  r2: RectType,
  num = 2
): void {
  expect(r1.x1).toBeCloseTo(r2.x1, num)
  expect(r1.x2).toBeCloseTo(r2.x2, num)
  expect(r1.y1).toBeCloseTo(r2.y1, num)
  expect(r1.y2).toBeCloseTo(r2.y2, num)
}

/**
 * Spawn a temporary folder with the following project structure:
 * 'test-fs-data/myProject': {
 * '.config': 'config contents',
 * 'project.json': 'project contents',
 * 'tasks': {
 * '000000.json': '{"testField": "testValue"}',
 * '000001.json': 'content1'
 * }
 * }
 * This is necessary because mock-fs doesn't implement the withFileTypes
 * option of fs.readDir correctly; and has flakiness issues
 *
 * @param dataDir
 * @param projectName
 */
export function makeProjectDir(dataDir: string, projectName: string): void {
  const projectDir = path.join(dataDir, StorageStructure.PROJECT, projectName)
  const taskDir = path.join(projectDir, "tasks")
  fs.ensureDirSync(projectDir)
  fs.ensureDirSync(taskDir)
  fs.writeFileSync(path.join(projectDir, ".config"), "config contents")
  fs.writeFileSync(path.join(projectDir, "project.json"), "project contents")
  const content0 = JSON.stringify({ testField: "testValue" })
  fs.writeFileSync(path.join(taskDir, "000000.json"), content0)
  fs.writeFileSync(path.join(taskDir, "000001.json"), "content1")
}

/**
 * The initial backend task represents the saved data
 *
 * @param sessionId
 */
export function getInitialState(sessionId: string): State {
  const partialTask: Partial<TaskType> = {
    items: [makeItem({ index: 0, id: "0" }, true)],
    sensors: { 0: makeSensor(0, "", "") }
  }
  const defaultTask = makeTask(partialTask)
  const defaultState = makeState({
    task: defaultTask
  })
  defaultState.session.id = sessionId
  defaultState.task.config.autosave = true
  return defaultState
}

/**
 * Helper function to get box2d actions
 *
 * @param itemIndex
 */
export function getRandomBox2dAction(itemIndex: number = 0): AddLabelsAction {
  return addBox2dLabel(
    itemIndex,
    0,
    [],
    {},
    {
      x1: Math.random(),
      y1: Math.random(),
      x2: Math.random(),
      y2: Math.random()
    }
  )
}

/**
 * Helper function to generate points of a polygon
 * In the format returned by the model server
 */
export function getRandomModelPoly(): number[][] {
  const points = []
  for (let i = 0; i < 5; i += 1) {
    points.push([Math.random(), Math.random()])
  }
  return points
}

/**
 * Get the path to the test config
 */
export function getTestConfigPath(): string {
  return "./app/config/test_config.yml"
}

/**
 * Load the test config
 */
export function getTestConfig(): ServerConfig {
  const configPath = getTestConfigPath()
  return parseConfig(configPath)
}

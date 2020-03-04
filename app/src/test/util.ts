import * as fs from 'fs-extra'
import * as path from 'path'
import { addBox2dLabel } from '../js/action/box2d'
import { makeItem,
  makeSensor, makeState, makeTask } from '../js/functional/states'
import { RectType, State, TaskType, Vector3Type } from '../js/functional/types'

/**
 * Check equality between two Vector3Type objects
 * @param v1
 * @param v2
 */
export function expectVector3TypesClose (
  v1: Vector3Type, v2: Vector3Type, num = 2
) {
  expect(v1.x).toBeCloseTo(v2.x, num)
  expect(v1.y).toBeCloseTo(v2.y, num)
  expect(v1.z).toBeCloseTo(v2.z, num)
}

/**
 * Check that rectangles are close
 */
export function expectRectTypesClose (
  r1: RectType, r2: RectType, num = 2
) {
  expect(r1.x1).toBeCloseTo(r2.x1, num)
  expect(r1.x2).toBeCloseTo(r2.x2, num)
  expect(r1.y1).toBeCloseTo(r2.y1, num)
  expect(r1.y2).toBeCloseTo(r2.y2, num)
}

/**
 * Spawn a temporary folder with the following project structure:
 *  'test-fs-data/myProject': {
 *     '.config': 'config contents',
 *    'project.json': 'project contents',
 *     'tasks': {
 *      '000000.json': '{"testField": "testValue"}',
 *       '000001.json': 'content1'
 *     }
 *   }
 * This is necessary because mock-fs doesn't implement the withFileTypes
 *   option of fs.readDir correctly; and has flakiness issues
 */
export function makeProjectDir (dataDir: string, projectName: string) {
  const projectDir = path.join(dataDir, projectName)
  const taskDir = path.join(projectDir, 'tasks')
  fs.ensureDirSync(projectDir)
  fs.ensureDirSync(taskDir)
  fs.writeFileSync(path.join(projectDir, '.config'), 'config contents')
  fs.writeFileSync(path.join(projectDir, 'project.json'), 'project contents')
  const content0 = JSON.stringify({ testField: 'testValue' })
  fs.writeFileSync(path.join(taskDir, '000000.json'), content0)
  fs.writeFileSync(path.join(taskDir, '000001.json'), 'content1')
}

/**
 * The initial backend task represents the saved data
 */
export function getInitialState (sessionId: string): State {
  const partialTask: Partial<TaskType> = {
    items: [makeItem({ id: 0 })],
    sensors: { 0: makeSensor(0, '', '') }
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
 */
export function getRandomBox2dAction () {
  return addBox2dLabel(0, 0, [], {},
    Math.random(), Math.random(), Math.random(), Math.random())
}

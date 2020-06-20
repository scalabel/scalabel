import * as fs from 'fs-extra'
import * as yaml from 'js-yaml'
import _ from 'lodash'
import * as path from 'path'
import { addBox2dLabel } from '../../../js/action/box2d'
import { makeItem,
  makeSensor, makeState, makeTask } from '../../../js/functional/states'
import { IdType, LabelIdMap,
  PolyPathPoint2DType, RectType, State, TaskType, Vector3Type } from '../../../js/functional/types'
import * as defaults from '../../../js/server/defaults'
import { ServerConfig } from '../../../js/server/types'

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
 * Check that the path point has the correct field values
 */
export function checkPathPointFields (
  point: PolyPathPoint2DType, x: number, y: number, isVertexType: boolean) {
  expect(point.x).toBe(x)
  expect(point.y).toBe(y)
  if (isVertexType) {
    expect(point.pointType).toBe('vertex')
  } else {
    expect(point.pointType).toBe('bezier')
  }
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
    items: [makeItem({ index: 0, id: '0' }, true)],
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
export function getRandomBox2dAction (itemIndex: number = 0) {
  return addBox2dLabel(itemIndex, 0, [], {},
    Math.random(), Math.random(), Math.random(), Math.random())
}

/**
 * Helper function to generate points of a polygon
 * In the format returned by the model server
 */
export function getRandomModelPoly () {
  const points = []
  for (let i = 0; i++; i < 5) {
    points.push([Math.random(), Math.random()])
  }
  return points
}

/**
 * Get the path to the test config
 */
export function getTestConfigPath (): string {
  return './app/config/test_config.yml'
}

/**
 * Load the test config
 */
export function getTestConfig (): ServerConfig {
  const configDir = getTestConfigPath()
  const testConfig = yaml.load(fs.readFileSync(configDir, 'utf8'))
  const fullConfig = {
    ...defaults.serverConfig,
    ...testConfig
  }
  return fullConfig
}

/**
 * Find the new label that is not already in the labelIds
 * @param labels
 * @param labelIds
 */
export function findNewLabels (
    labels: LabelIdMap, labelIds: IdType[]): IdType[] {
  return _.filter(
    _.keys(labels),
    (id) => !labelIds.includes(id))
}

/**
 * Find the new label that is not already in the labelIds
 * @param labels
 * @param labelIds
 */
export function findNewLabelsFromState (
    state: State, itemIndex: number, labelIds: IdType[]): IdType[] {
  const labels = state.task.items[itemIndex].labels
  return _.filter(
  _.keys(labels),
  (id) => !labelIds.includes(id))
}

/**
 * Find the new label that is not already in the labelIds
 * @param labels
 * @param labelIds
 */
export function findNewTracksFromState (
  state: State, trackIds: IdType[]): IdType[] {
  const tracks = state.task.tracks
  return _.filter(
    _.keys(tracks),
  (id) => !trackIds.includes(id))
}

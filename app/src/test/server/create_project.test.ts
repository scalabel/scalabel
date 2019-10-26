import mockfs from 'mock-fs'
import { State, TaskType } from '../../js/functional/types'
import { createProject, createTasks,
  saveProject, saveTasks } from '../../js/server/create_project'
import { convertStateToExport } from '../../js/server/export'
import Session from '../../js/server/server_session'
import { CreationForm, Project } from '../../js/server/types'
import { getProjectKey, getTaskKey, initStorage } from '../../js/server/util'
import { sampleFormFileData, sampleFormImage,
  sampleFormVideo, sampleProjectAutolabel, sampleProjectImage,
  sampleProjectVideo, sampleTasksImage } from '../test_creation_objects'
import { sampleStateExport } from '../test_export_objects'

beforeAll(() => {
  // mock the file system for saving/loading
  mockfs({
    'data/': {}
  })

  // init global env to default
  const defaultEnv = {}
  Session.setEnv(defaultEnv)
  // init global storage
  initStorage(Session.getEnv())
})

// TODO- test that form is loaded correctly

describe('test project.json creation', () => {
  test('image project creation', () => {
    return testProjectCreation(sampleFormImage, sampleProjectImage)
  })

  test('video project creation', () => {
    return testProjectCreation(sampleFormVideo, sampleProjectVideo)
  })

  test('image project saving', () => {
    return testProjectSaving(sampleProjectImage)
  })

  test('video project saving', () => {
    return testProjectSaving(sampleProjectVideo)
  })
})

describe('test task.json creation', () => {
  test('task creation', () => {
    return createTasks(sampleProjectImage).then((tasks) => {
      expect(tasks).toEqual(sampleTasksImage)
    })
  })

  test('task saving', () => {
    return testTaskSaving(sampleTasksImage)
  })
})

describe('create with auto labels', () => {
  test('import then export', () => {
    return createTasks(sampleProjectAutolabel).then((tasks) => {
      // only 1 task should be created
      const state: Partial<State> = {
        task: tasks[0]
      }
      const exportedItems = convertStateToExport(state as State, 0)
      expect(exportedItems).toEqual(sampleStateExport)
    })
  })
})

/**
 * Tested that desired project is created from form
 */
async function testProjectCreation (
  sampleForm: CreationForm, sampleProject: Project): Promise<void> {
  return createProject(sampleForm, sampleFormFileData).then((project) => {
    expect(project).toEqual(sampleProject)
    return
  })
}

/**
 * Tests that project is saved correctly
 */
async function testProjectSaving (sampleProject: Project): Promise<void> {
  await saveProject(sampleProject)

  // check that it saved by loading it
  const key = getProjectKey(sampleProject.config.projectName)
  const loadedProjectData = await Session.getStorage().load(key)

  // make sure what's loaded is what was saved
  const loadedProject = JSON.parse(loadedProjectData)
  expect(loadedProject).toEqual(sampleProject)
}

/**
 * Tests that task is saved correctly
 */
async function testTaskSaving (sampleTasks: TaskType[]): Promise<void> {
  await saveTasks(sampleTasks)

  // check that tasks saved by loading them
  for (const task of sampleTasks) {
    const key = getTaskKey(task.config.projectName, task.config.taskId)
    const loadedTaskData = await Session.getStorage().load(key)
    // make sure what's loaded is what was saved
    const loadedTask = JSON.parse(loadedTaskData)
    expect(loadedTask).toEqual(task)
  }
}

afterAll(() => {
  mockfs.restore()
})

import mockfs from 'mock-fs'
import { State, TaskType } from '../../js/functional/types'
import { createProject, createTasks,
  saveProject, saveTasks } from '../../js/server/create_project'
import { convertStateToExport } from '../../js/server/export'
import Session from '../../js/server/server_session'
import { CreationForm, FormFileData, Project } from '../../js/server/types'
import { getProjectKey, getTaskKey, initStorage } from '../../js/server/util'
import {
  sampleFormFileData,
  sampleFormImage,
  sampleFormVideo,
  sampleProjectAutolabel,
  sampleProjectAutolabelPolygon,
  sampleProjectImage,
  sampleProjectVideo,
  sampleTasksImage,
  sampleTasksVideo,
  sampleVideoFormFileData
} from '../test_creation_objects'
import { sampleStateExportImage, sampleStateExportImagePolygon } from '../test_export_objects'

beforeAll(async () => {
  // mock the file system for saving/loading
  mockfs({
    'data/': {}
  })

  // init global env to default
  const defaultEnv = {}
  Session.setEnv(defaultEnv)
  // init global storage
  await initStorage(Session.getEnv())
})

// TODO- test that form is loaded correctly

describe('test project.json creation', () => {
  test('image project creation', () => {
    return testProjectCreation(
      sampleFormImage, sampleProjectImage, sampleFormFileData
    )
  })

  test('video project creation', () => {
    return testProjectCreation(
      sampleFormVideo, sampleProjectVideo, sampleVideoFormFileData
    )
  })

  test('image project saving', () => {
    return testProjectSaving(sampleProjectImage)
  })

  test('video project saving', () => {
    return testProjectSaving(sampleProjectVideo)
  })
})

describe('test task.json creation', () => {
  test('task non-tracking creation', () => {
    return createTasks(sampleProjectImage).then((tasks) => {
      expect(tasks).toEqual(sampleTasksImage)
    })
  })

  test('test tracking creation', async () => {
    return createTasks(sampleProjectVideo).then((tasks) => {
      expect(tasks).toEqual(sampleTasksVideo)
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
      const exportedItems = convertStateToExport(state as State)
      expect(exportedItems).toEqual(sampleStateExportImage)
    })
  })
  test('import then export for polygon', () => {
    return createTasks(sampleProjectAutolabelPolygon).then((tasks) => {
      // only 1 task should be created
      const state: Partial<State> = {
        task: tasks[0]
      }
      const exportedItems = convertStateToExport(state as State)
      expect(exportedItems).toEqual(sampleStateExportImagePolygon)
    })
  })
})

/**
 * Tested that desired project is created from form
 */
async function testProjectCreation (
  sampleForm: CreationForm,
  sampleProject: Project,
  formFileData: FormFileData
): Promise<void> {
  return createProject(sampleForm, formFileData).then((project) => {
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

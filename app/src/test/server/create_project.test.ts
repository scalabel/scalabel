import mockfs from 'mock-fs'
import { State, TaskType } from '../../js/functional/types'
import { createProject, createTasks } from '../../js/server/create_project'
import { convertStateToExport } from '../../js/server/export'
import { FileStorage } from '../../js/server/file_storage'
import { ProjectStore } from '../../js/server/project_store'
import {
  CreationForm, FormFileData, Project
} from '../../js/server/types'
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
import {
  sampleStateExportImage, sampleStateExportImagePolygon
} from '../test_export_objects'

let projectStore: ProjectStore

beforeAll(async () => {
  // mock the file system for saving/loading
  mockfs({
    'data/': {}
  })

  const storage = new FileStorage('data')
  projectStore = new ProjectStore(storage)
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
  await projectStore.saveProject(sampleProject)

  // check that it saved correctly by loading it and comparing
  const loadedProject = await projectStore.loadProject(
    sampleProject.config.projectName)

  expect(loadedProject).toEqual(sampleProject)
}

/**
 * Tests that task is saved correctly
 */
async function testTaskSaving (sampleTasks: TaskType[]): Promise<void> {
  await projectStore.saveTasks(sampleTasks)

  // check that tasks saved correctly by loading them and comparing
  for (const task of sampleTasks) {
    const projectName = task.config.projectName
    const taskId = task.config.taskId
    const loadedTask = await projectStore.loadTask(projectName, taskId)
    expect(loadedTask).toEqual(task)
  }
}

afterAll(() => {
  mockfs.restore()
})

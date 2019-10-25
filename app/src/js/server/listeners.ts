import {
  NextFunction,
  Request,
  Response
} from 'express'
import { sprintf } from 'sprintf-js'
import * as uuid4 from 'uuid/v4'
import { ItemTypeName, LabelTypeName, TrackPolicyType } from '../common/types'
import { DashboardContents, ProjectOptions, TaskOptions } from '../components/dashboard'
import { ItemExport } from '../functional/bdd_types'
import { makeState } from '../functional/states'
import { State, TaskType } from '../functional/types'
import {
  createProject, createTasks, parseFiles,
  parseForm, saveProject, saveTasks
} from './create_project'
import { convertStateToExport } from './export'
import { getExportName } from './path'
import Session from './server_session'
import * as types from './types'
import { getExistingProjects, getProjectKey, getTaskKey, getTasksInProject, index2str, loadSavedState, logError, logInfo } from './util'

/**
 * Logs requests to static or dynamic files
 */
export function LoggingHandler (
  req: Request, _res: Response, next: NextFunction) {
  const log = sprintf('Requesting %s', req.originalUrl)
  logInfo(log)
  next()
}

/**
 * Handles getting all projects' names
 */
export async function ProjectNameHandler (_req: Request, res: Response) {
  let projects: string[]
  const defaultProjects = ['No existing project']
  try {
    projects = await getExistingProjects()
    if (projects.length === 0) {
      projects = defaultProjects
    }
  } catch (err) {
    logError(err)
    projects = defaultProjects
  }
  const projectNames = JSON.stringify(projects)
  res.send(projectNames)
  res.end()
}

/**
 * Handles posting export
 * @param req
 * @param res
 */
export async function GetExportHandler (req: Request, res: Response) {
  if (req.method !== 'GET' || req.query === {}) {
    res.sendStatus(404)
    res.end()
  }
  try {
    const projectName = req.query[types.FormField.PROJECT_NAME]
    const key = getProjectKey(projectName)
    const fields = await Session.getStorage().load(key)
    const projectToLoad = JSON.parse(fields) as types.Project
    // grab the latest submissions from all tasks
    const tasks = await getTasksInProject(projectName)
    let items: ItemExport[] = []
    // load the latest submission for each task to export
    for (const task of tasks) {
      await loadSavedState(projectName, task)
        .then((state: State) => {
          items = items.concat(convertStateToExport(state))
        })
        .catch((err: Error) => {
          // if state submission is not found, use an empty item
          logInfo(err.message)
          for (const itemToLoad of task.items) {
            items.push({
              name: itemToLoad.url,
              url: itemToLoad.url,
              videoName: '',
              timestamp: projectToLoad.config.submitTime,
              attributes: {},
              index: itemToLoad.index,
              labels: []
            })
          }
        })
    }
    const exportJson = JSON.stringify(items, null, '  ')
    // set relevant header and send the exported json file
    res.attachment(getExportName(projectName))
    res.end(Buffer.from(exportJson, 'binary'), 'binary')
  } catch (err) {
    logError(err)
    res.end()
  }
}

/**
 * Handles posted project form data
 */
export async function PostProjectHandler (req: Request, res: Response) {
  if (req.method !== 'POST' || req.fields === undefined) {
    res.sendStatus(404)
  }
  const fields = req.fields
  const files = req.files
  if (fields !== undefined && files !== undefined) {
    try {
      // parse form from request
      const form = await parseForm(fields)
      // parse item, category, and attribute data from the form
      const formFileData = await parseFiles(form.labelType, files)
      // create the project from the form data
      const project = await createProject(form, formFileData)
      await Promise.all([
        // save the project
        saveProject(project),
        // create tasks then save them
        createTasks(project).then(
          (tasks: TaskType[]) => saveTasks(tasks))
      ])
      res.send()
    } catch (err) {
      logError(err)
      // alert the user that something failed
      res.send(err.message)
    }
  } else {
    // alert the user that the sent fields were illegal
    const err = Error('illegal fields')
    logError(err)
    res.send(err.message)
  }
}

/**
 * Return dashboard info
 * @param req
 * @param res
 */
export async function DashboardHandler (req: Request, res: Response) {
  if (req.method !== 'POST') {
    res.sendStatus(404)
    res.end()
    return
  }

  const body = req.body
  if (body) {
    try {
      const name = body.name
      const key = getProjectKey(name)
      const fields = await Session.getStorage().load(key)
      const project = JSON.parse(fields) as types.Project
      // grab the latest submissions from all tasks
      const tasks = await getTasksInProject(name)
      const projectOptions: ProjectOptions = {
        name: project.config.projectName,
        itemType: project.config.itemType,
        labelTypes: project.config.labelTypes,
        taskSize: project.config.taskSize,
        numItems: project.items.length,
        numLeafCategories: project.config.categories.length,
        numAttributes: project.config.attributes.length
      }

      const taskOptions = []
      for (const task of tasks) {
        let numLabeledItems = 0
        let numLabels = 0
        for (const item of task.items) {
          const currNumLabels = Object.keys(item.labels).length
          if (item.labels && currNumLabels > 0) {
            numLabeledItems++
            numLabels += currNumLabels
          }
        }

        const options: TaskOptions = {
          numLabeledItems: numLabeledItems.toString(),
          numLabels: numLabels.toString(),
          submitted: task.config.submitted,
          handlerUrl: task.config.handlerUrl
        }

        taskOptions.push(options)
      }

      const contents: DashboardContents = {
        projectMetaData: projectOptions,
        taskMetaDatas: taskOptions
      }

      res.send(JSON.stringify(contents))
    } catch (err) {
      logError(err)
      res.send(err.message)
    }
  }
}

/**
 * Handler for loading tasks
 * @param req
 * @param res
 */
export async function LoadHandler (req: Request, res: Response) {
  if (req.method !== 'POST') {
    res.sendStatus(404)
    res.end()
    return
  }

  const body = req.body
  if (body) {
    try {
      const name = body.task.projectOptions.name
      const index = body.task.index
      const key = getTaskKey(name, index2str(index))
      const fields = await Session.getStorage().load(key)
      const task = JSON.parse(fields) as TaskType
      const state = makeState({ task })

      state.session.items = state.task.items.map((_i) => ({ loaded: false }))

      switch (state.task.config.itemType) {
        case ItemTypeName.IMAGE:
        case ItemTypeName.VIDEO:
          if (state.task.config.labelTypes.length === 1 &&
              state.task.config.labelTypes[0] === LabelTypeName.BOX_2D) {
            state.task.config.policyTypes =
              [TrackPolicyType.LINEAR_INTERPOLATION_BOX_2D]
          }
          break
        case ItemTypeName.POINT_CLOUD:
        case ItemTypeName.POINT_CLOUD_TRACKING:
          if (state.task.config.labelTypes.length === 1 &&
              state.task.config.labelTypes[0] === LabelTypeName.BOX_3D) {
            state.task.config.policyTypes =
              [TrackPolicyType.LINEAR_INTERPOLATION_BOX_3D]
          }
          break
      }

      state.session.id = uuid4()

      res.send(JSON.stringify(state))
    } catch (err) {
      logError(err)
      res.send(err.message)
    }
  }
}

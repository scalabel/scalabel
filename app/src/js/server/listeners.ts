import {
  NextFunction,
  Request,
  Response
} from 'express'
import { sprintf } from 'sprintf-js'
import { DashboardContents, ProjectOptions, TaskOptions } from '../components/dashboard'
import { ItemExport } from '../functional/bdd_types'
import { TaskType } from '../functional/types'
import {
  createProject, createTasks, parseFiles,
  parseForm, saveProject, saveTasks
} from './create_project'
import { convertStateToExport } from './export'
import { loadState } from './hub'
import Logger from './logger'
import { getExportName } from './path'
import { RedisCache } from './redis_cache'
import * as types from './types'
import { getExistingProjects, getTasksInProject, loadProject } from './util'

/**
 * Logs requests to static or dynamic files
 */
export function LoggingHandler (
  req: Request, _res: Response, next: NextFunction) {
  const log = sprintf('Requesting %s', req.originalUrl)
  Logger.info(log)
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
    Logger.error(err)
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
    const loadedProject = await loadProject(projectName)
    // grab the latest submissions from all tasks
    const tasks = await getTasksInProject(projectName)
    let items: ItemExport[] = []
    // load the latest submission for each task to export
    for (const task of tasks) {
      try {
        const cache = new RedisCache()
        const state = await loadState(projectName, task.config.taskId, cache)
        items = items.concat(convertStateToExport(state))
      } catch (error) {
        Logger.info(error.message)
        for (const itemToLoad of task.items) {
          const url = Object.values(itemToLoad.urls)[0]
          items.push({
            name: url,
            url,
            sensor: -1,
            timestamp: loadedProject.config.submitTime,
            videoName: itemToLoad.videoName,
            attributes: {},
            labels: []
          })
        }
      }
    }
    const exportJson = JSON.stringify(items, null, '  ')
    // set relevant header and send the exported json file
    res.attachment(getExportName(projectName))
    res.end(Buffer.from(exportJson, 'binary'), 'binary')
  } catch (error) {
    // TODO: Be more specific about what this error may be
    Logger.error(error)
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
        saveProject(project),
        // create tasks then save them
        createTasks(project).then(
          (tasks: TaskType[]) => saveTasks(tasks))
        // save the project
      ])
      res.send()
    } catch (err) {
      Logger.error(err)
      // alert the user that something failed
      res.send(err.message)
    }
  } else {
    // alert the user that the sent fields were illegal
    const err = Error('illegal fields')
    Logger.error(err)
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
      const projectName = body.name
      const project = await loadProject(projectName)
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
      for (const emptyTask of tasks) {
        let task
        try {
          // first, attempt loading previous submission
          // TODO: Load the previous state asynchronously in dashboard
          const cache = new RedisCache()
          const state = await loadState(projectName,
              emptyTask.config.taskId, cache)
          task = state.task
        } catch {
          task = emptyTask
        }
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
      Logger.error(err)
      res.send(err.message)
    }
  }
}

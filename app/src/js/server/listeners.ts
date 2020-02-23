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
  createProject, createTasks, parseFiles, parseForm
} from './create_project'
import { convertStateToExport } from './export'
import Logger from './logger'
import { getExportName } from './path'
import { ProjectStore } from './project_store'
import * as types from './types'
import { UserManager } from './user_manager'
import { countLabels } from './util'

/**
 * Wraps HTTP listeners
 */
export class Listeners {
  /** the project store */
  protected projectStore: ProjectStore
  /** the user manager */
  protected userManager: UserManager

  constructor (projectStore: ProjectStore, userManager: UserManager) {
    this.projectStore = projectStore
    this.userManager = userManager
  }

  /**
   * Logs requests to static or dynamic files
   */
  public loggingHandler (
    req: Request, _res: Response, next: NextFunction) {
    const log = sprintf('Requesting %s', req.originalUrl)
    Logger.info(log)
    next()
  }

  /**
   * Handles getting all projects' names
   */
  public async projectNameHandler (_req: Request, res: Response) {
    let projects: string[]
    const defaultProjects = ['No existing project']
    try {
      projects = await this.projectStore.getExistingProjects()
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
   */
  public async getExportHandler (req: Request, res: Response) {
    if (req.method !== 'GET' || req.query === {}) {
      res.sendStatus(404)
      res.end()
    }
    try {
      const projectName = req.query[types.FormField.PROJECT_NAME]
      // grab the latest submissions from all tasks
      const tasks = await this.projectStore.getTasksInProject(projectName)
      let items: ItemExport[] = []
      // load the latest submission for each task to export
      for (const task of tasks) {
        try {
          const taskId = task.config.taskId
          const state = await this.projectStore.loadState(projectName, taskId)
          items = items.concat(convertStateToExport(state))
        } catch (error) {
          Logger.info(error.message)
          for (const itemToLoad of task.items) {
            const url = Object.values(itemToLoad.urls)[0]
            let timestamp = -1
            const submissions = task.progress.submissions
            if (submissions.length > 0) {
              const latestSubmission = submissions[submissions.length - 1]
              timestamp = latestSubmission.time
            }
            items.push({
              name: url,
              url,
              sensor: -1,
              timestamp,
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
  public async postProjectHandler (req: Request, res: Response) {
    if (req.method !== 'POST' || req.fields === undefined) {
      res.sendStatus(404)
    }
    const fields = req.fields
    const files = req.files
    if (fields !== undefined && files !== undefined) {
      try {
        // parse form from request
        const form = await parseForm(fields, this.projectStore)
        // parse item, category, and attribute data from the form
        const formFileData = await parseFiles(form.labelType, files)
        // create the project from the form data
        const project = await createProject(form, formFileData)
        await Promise.all([
          this.projectStore.saveProject(project),
          // create tasks then save them
          createTasks(project).then(
            (tasks: TaskType[]) => this.projectStore.saveTasks(tasks))
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
   */
  public async dashboardHandler (req: Request, res: Response) {
    if (req.method !== 'POST') {
      res.sendStatus(404)
      res.end()
      return
    }

    const body = req.body
    if (body) {
      try {
        const projectName = body.name
        const project = await this.projectStore.loadProject(projectName)
        // grab the latest submissions from all tasks
        const tasks = await this.projectStore.getTasksInProject(projectName)
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
          let task: TaskType
          try {
            // first, attempt loading previous submission
            // TODO: Load the previous state asynchronously in dashboard
            const taskId = emptyTask.config.taskId
            const state = await this.projectStore.loadState(projectName, taskId)
            task = state.task
          } catch {
            task = emptyTask
          }
          const [numLabeledItems, numLabels] = countLabels(task)
          const submitted = task.progress.submissions.length > 0
          const options: TaskOptions = {
            numLabeledItems: numLabeledItems.toString(),
            numLabels: numLabels.toString(),
            submitted,
            handlerUrl: task.config.handlerUrl
          }

          taskOptions.push(options)
        }

        const numUsers = await this.userManager.countUsers(projectOptions.name)
        const contents: DashboardContents = {
          projectMetaData: projectOptions,
          taskMetaDatas: taskOptions,
          numUsers
        }

        res.send(JSON.stringify(contents))
      } catch (err) {
        Logger.error(err)
        res.send(err.message)
      }
    }
  }
}

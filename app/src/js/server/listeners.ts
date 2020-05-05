import {
  NextFunction,
  Request,
  Response
} from 'express'
import { File } from 'formidable'
import { sprintf } from 'sprintf-js'
import { DashboardContents, ProjectOptions, TaskOptions } from '../components/dashboard'
import { getSubmissionTime } from '../components/util'
import { TaskType } from '../functional/types'
import { ItemExport } from './bdd_types'
import {
  createProject, createTasks, parseFiles, parseForm, readItemsFile
} from './create_project'
import { convertStateToExport } from './export'
import Logger from './logger'
import { getExportName } from './path'
import { ProjectStore } from './project_store'
import * as types from './types'
import { UserManager } from './user_manager'
import { countLabels, parseProjectName } from './util'

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
      const projectName = req.query[types.FormField.PROJECT_NAME] as string
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
            const timestamp = getSubmissionTime(task.progress.submissions)
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
   * Alert the user that the sent fields were illegal
   */
  public badFormResponse (res: Response) {
    const err = Error('Illegal fields for project creation')
    Logger.error(err)
    res.status(400).send(err.message)
  }

  /**
   * Alert the user that the task creation request was illegal
   */
  public badTaskResponse (res: Response) {
    const err = Error('Illegal fields for task creation')
    Logger.error(err)
    res.status(400).send(err.message)
  }

  /**
   * Error if it's not a post request
   */
  public checkInvalidPost (req: Request, res: Response): boolean {
    if (req.method !== 'POST') {
      res.sendStatus(404)
      res.end()
      return true
    }
    return false
  }

  /**
   * Handles posted project from internal data
   * Items file not required
   */
  public async postProjectInternalHandler (req: Request, res: Response) {
    if (this.checkInvalidPost(req, res)) {
      return
    }

    if (req.body === undefined ||
        req.body.fields === undefined ||
        req.body.files === undefined) {
      this.badFormResponse(res)
      return
    }

    await this.createProjectFromDicts(
      req.body.fields, req.body.files, false, res)
  }

  /**
   * Handles posted project from form data
   * Items file required
   */
  public async postProjectHandler (req: Request, res: Response) {
    if (this.checkInvalidPost(req, res)) {
      return
    }

    if (req.fields === undefined || req.files === undefined) {
      this.badFormResponse(res)
      return
    }

    const fields = req.fields as { [key: string]: string}
    const formFiles = req.files as { [key: string]: File | undefined }
    const files: { [key: string]: string } = {}
    for (const key of Object.keys(formFiles)) {
      const file = formFiles[key]
      if (file !== undefined && file.size !== 0) {
        files[key] = file.path
      }
    }

    await this.createProjectFromDicts(fields, files, true, res)
  }

  /**
   * Handles tasks being added to a project
   */
  public async postTasksHandler (req: Request, res: Response) {
    if (this.checkInvalidPost(req, res)) {
      return
    }

    if (req.body === undefined
      || req.body.items === undefined
      || req.body.projectName === undefined) {
      this.badTaskResponse(res)
      return
    }

    // read in the data
    const items = await readItemsFile(req.body.items)
    let project: types.Project
    let projectName: string
    try {
      projectName = parseProjectName(req.body.projectName)
      project = await this.projectStore.loadProject(projectName)
    } catch (err) {
      Logger.error(err)
      this.badTaskResponse(res)
      return
    }

    // update the project with the new items
    const itemStartNum = project.items.length
    project.items = project.items.concat(items)
    await this.projectStore.saveProject(project)

    // update the tasks, make sure not to combine old and new items
    project.items = items
    const oldTasks = await this.projectStore.getTasksInProject(projectName)
    const taskStartNum = oldTasks.length
    const tasks = await createTasks(project, taskStartNum, itemStartNum)
    await this.projectStore.saveTasks(tasks)

    res.sendStatus(200)
  }

  /**
   * Return dashboard info
   */
  public async dashboardHandler (req: Request, res: Response) {
    if (this.checkInvalidPost(req, res)) {
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
          const options: TaskOptions = {
            numLabeledItems: numLabeledItems.toString(),
            numLabels: numLabels.toString(),
            submissions: task.progress.submissions,
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

  /**
   * Finishes project creation using processed dicts
   */
  private async createProjectFromDicts (
    fields: { [key: string]: string },
    files: { [key: string]: string },
    itemsRequired: boolean, res: Response) {
    try {
        // parse form from request
      const form = await parseForm(fields, this.projectStore)
        // parse item, category, and attribute data from the form
      const formFileData = await parseFiles(
        form.labelType, files, itemsRequired)
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
      res.status(400).send(err.message)
    }
  }
}

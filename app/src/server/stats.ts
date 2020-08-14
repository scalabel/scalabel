import _ from 'lodash'
import { ProjectOptions, TaskOptions } from '../components/dashboard'
import { Project } from '../types/project'
import { TaskType } from '../types/state'

/**
 * Extract ProjectOption from a Project
 */
export function getProjectOptions (project: Project): ProjectOptions {
  return {
    name: project.config.projectName,
    itemType: project.config.itemType,
    labelTypes: project.config.labelTypes,
    taskSize: project.config.taskSize,
    numItems: project.items.length,
    numLeafCategories: project.config.categories.length,
    numAttributes: project.config.attributes.length
  }
}

/**
 * Returns the total number of labels in the task
 */
export function countLabelsTask (task: TaskType): number {
  const numPerItem = task.items.map((item) => _.size(item.labels))
  return _.sum(numPerItem)
}

/**
 * Returns the number of items with at least 1 label in the task
 */
export function countLabeledItemsTask (task: TaskType): number {
  const labeledItems = task.items.filter((item) => _.size(item.labels) > 0)
  return labeledItems.length
}

/**
 * Returns the total number of labels in the project
 */
export function countLabelsProject (tasks: TaskType[]): number {
  return _.sum(tasks.map(countLabelsTask))
}

/**
 * Returns the number of items with at least 1 label in the project
 */
export function countLabeledItemsProject (tasks: TaskType[]): number {
  return _.sum(tasks.map(countLabeledItemsTask))

}

/**
 * Extracts TaskOptions from a Task
 */
export function getTaskOptions (task: TaskType): TaskOptions {
  const numLabeledItems = countLabeledItemsTask(task)
  const numLabels = countLabelsTask(task)
  return {
    numLabeledItems: numLabeledItems.toString(),
    numLabels: numLabels.toString(),
    submissions: task.progress.submissions,
    handlerUrl: task.config.handlerUrl
  }
}

/**
 * Get the total number of items across all tasks
 */
export function getNumItems (tasks: TaskType[]) {
  const itemsPerTask = tasks.map((task) => task.items.length)
  return _.sum(itemsPerTask)
}

/**
 * Get the number of tasks with a submission
 */
export function getNumSubmissions (tasks: TaskType[]) {
  const submittedTasks = tasks.filter((task) =>
    task.progress.submissions.length > 0)
  return submittedTasks.length
}

/**
 * Get the number of labels with each category
 */
export function getCategoryCounts (tasks: TaskType[]) {
  // TODO: get real counts, get 0 even if category missing
  const countsByTask = tasks.map((_task) => {
    const dict: { [key: string]: number } = {}
    dict.a = 1
    return dict
  })
  const totalCounts: { [key: string]: number } = {}
  for (const taskCount of countsByTask) {
    for (const key of Object.keys(taskCount)) {
      if (!(key in totalCounts)) {
        totalCounts[key] = 0
      }
      totalCounts[key] += taskCount[key]
    }
  }
  return totalCounts
}

/**
 * {category: count}
 * [{attribute1Options: count}, ...]
 */
interface ProjectStats {
  /** the total number of labels */
  numLabels: number
  /** the number of labeled items */
  numLabeledItems: number
  /** the total number of items */
  numItems: number
  /** the number of submitted tasks */
  numSubmittedTasks: number
  /** the total number of tasks */
  numTasks: number
  /** map from category name to count */
  categoryCounts: { [key: string]: number }
}
/**
 * Get the stats for a collection of tasks from a project
 */
export function getProjectStats (tasks: TaskType[]): ProjectStats {
  return {
    numLabels: countLabelsProject(tasks),
    numLabeledItems: countLabeledItemsProject(tasks),
    numItems: getNumItems(tasks),
    numSubmittedTasks: getNumSubmissions(tasks),
    numTasks: tasks.length,
    categoryCounts: getCategoryCounts(tasks)
  }
}

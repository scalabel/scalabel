import { sprintf } from 'sprintf-js'
import { SubmitData } from '../functional/types'

/**
 * Puts Date.now into dashboard display format
 */
export function formatDate (dateNow: number): string {
  const date = new Date(dateNow)
  const day = date.toDateString()
  const time = date.toTimeString()
  return sprintf('%s %s', day, time)
}

/**
 * Extract timestamp of latest submission; -1 if not submitted
 */
export function getSubmissionTime (submissions: SubmitData[]) {
  if (submissions.length > 0) {
    const latestSubmission = submissions[submissions.length - 1]
    return latestSubmission.time
  }
  return -1
}

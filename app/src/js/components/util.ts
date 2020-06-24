import { SubmitData } from '../functional/types'

/**
 * Puts Date.now into dashboard display format
 */
export function formatDate (dateNow: number): string {
  const date = new Date(dateNow)
  return date.toLocaleString('en-CA', { hour12: false })
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

/**
 * Add callback for the main window visibility change
 */
export function addVisibilityListener (callback: (visible: boolean) => void) {
  window.addEventListener('blur', () => {
    callback(false)
  })
  window.addEventListener('focus', () => {
    callback(true)
  })
  window.addEventListener('visibilitychange', () => {
    callback(!document.hidden)
  })
}

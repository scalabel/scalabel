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

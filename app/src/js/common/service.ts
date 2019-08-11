/**
 * This function post request to backend to retrieve requried data to display
 * @param {string} url
 * @param {string} method
 * @param {boolean} async
 * @return {function} data
 */
export function requestData<DataType> (url: string,
                                       method: string,
                                       async: boolean): DataType[] {
  let data!: DataType[]
  const xhr = new XMLHttpRequest()
  xhr.onreadystatechange = () => {
    if (xhr.readyState === 4 && xhr.status === 200) {
      data = JSON.parse(xhr.responseText)
    }
  }
  xhr.open(method, url, async)
  xhr.send(null)
  return data
}

/**
 * This function post request to backend to retrieve users' information
 * @return {function} projects
 */
export function getProjects (): string[] {
  return requestData('./postProjectNames', 'get', false)
}

/**
 * Redirect user to create new projects
 */
export function goCreate (): void {
  window.location.href = '/create'
}

/**
 * Redirect user to logOut page
 */
export function logout (): void {
  window.location.href = '/logOut'
}

/**
 * Redirect user(either admin or worker) to the project's dashboard
 * @param {string} projectName - the values to convert.
 */
export function toProject (projectName: string): void {
  window.location.href = '/dashboard?project_name=' + projectName
}

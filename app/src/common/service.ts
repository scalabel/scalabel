import { Endpoint } from "../const/connection"
import Cognito from "./cognito"

/**
 * This function post request to backend to retrieve required data to display
 *
 * @param {string} url
 * @param {string} method
 * @param {boolean} async
 * @returns {Function} data
 */
export function requestData<DataType>(
  url: string,
  method: string,
  async: boolean
): DataType[] {
  let data!: DataType[]
  const xhr = new XMLHttpRequest()
  xhr.onreadystatechange = () => {
    if (xhr.readyState === 4) {
      if (xhr.status === 200) {
        data = JSON.parse(xhr.responseText)
      } else if (xhr.status === 401) {
        const res = JSON.parse(xhr.responseText)
        window.location.href = res.redirect
      }
    }
  }
  xhr.open(method, url, async)
  const auth = getAuth()
  if (auth !== "") {
    xhr.setRequestHeader("Authorization", auth)
  }
  xhr.send(null)
  return data
}

/**
 * Get auth status
 *
 * @export
 * @returns
 */
export function getAuth(): string {
  // Check auth info
  return Cognito.getAuth()
}

/**
 * This function get request to backend to retrieve users' information
 *
 * @returns {string[]} projects
 */
export function getProjects(): string[] {
  return requestData(Endpoint.GET_PROJECT_NAMES, "get", false)
}

/**
 * Redirect user to create new projects
 */
export function goCreate(): void {
  window.location.href = "/create"
}

/**
 * Redirect user to logOut page
 */
export function logout(): void {
  window.location.href = "/logOut"
}

/**
 * Redirect user(either admin or worker) to the project's dashboard
 *
 * @param {string} projectName - the values to convert.
 */
export function toProject(projectName: string): void {
  window.location.href = "/dashboard?project_name=" + projectName
}

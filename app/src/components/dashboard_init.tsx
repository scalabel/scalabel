import { CssBaseline } from "@material-ui/core"
import { ThemeProvider } from "@material-ui/styles"
import React from "react"
import ReactDOM from "react-dom"

import { getAuth } from "../common/service"
import { handleInvalidPage } from "../common/util"
import { QueryArg } from "../const/common"
import { Endpoint } from "../const/connection"
import { scalabelTheme } from "../styles/theme"
import Dashboard, { DashboardContents } from "./dashboard"

/**
 * This function post requests to backend to retrieve dashboard contents
 *
 * @param vendor
 */
export function initDashboard(vendor: boolean): void {
  let dashboardContents: DashboardContents
  // Get params from url path.
  const searchParams = new URLSearchParams(window.location.search)
  const projectName = searchParams.get(QueryArg.PROJECT_NAME)
  if (projectName === null) {
    return handleInvalidPage()
  }

  const xhr = new XMLHttpRequest()
  xhr.onreadystatechange = () => {
    if (xhr.readyState === 4 && xhr.status === 200) {
      dashboardContents = JSON.parse(xhr.responseText)
      ReactDOM.render(
        <ThemeProvider theme={scalabelTheme}>
          <CssBaseline />
          <Dashboard dashboardContents={dashboardContents} vendor={vendor} />
        </ThemeProvider>,
        document.getElementById(vendor ? "vendor-root" : "dashboard-root")
      )
    }
  }

  xhr.open("GET", `${Endpoint.DASHBOARD}?name=${projectName}`)
  xhr.setRequestHeader("Content-Type", "application/json")
  const auth = getAuth()
  if (auth !== "") {
    xhr.setRequestHeader("Authorization", auth)
  }
  xhr.send()
}

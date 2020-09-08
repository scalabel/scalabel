import { ThemeProvider } from "@material-ui/core/styles"
import React from "react"
import ReactDOM from "react-dom"

import Dashboard from "../components/dashboard_admin"
import { scalabelTheme } from "../styles/theme"

ReactDOM.render(
  <ThemeProvider theme={scalabelTheme}>
    <Dashboard />
  </ThemeProvider>,
  document.getElementById("admin")
)

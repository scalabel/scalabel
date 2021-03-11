import { Divider } from "@material-ui/core"
import { withStyles } from "@material-ui/core/styles"
import Typography from "@material-ui/core/Typography"
import React from "react"

import { getProjects } from "../common/service"
import { dashboardStyles } from "../styles/dashboard"
import DashboardHeader from "./dashboard_header"
import DataTable from "./dashboard_table"
import HeaderPage from "./header_page"

export interface User {
  /** User ID */
  id: string
  /** User email */
  email: string
  /** User group */
  group: string
  /** User refresh token */
  refreshToken: string
  /** User's projects */
  projects: string[]
}

export interface DashboardClassType {
  /** root class */
  workerRoot: string
  /** spacing for app bar */
  appBarSpacer: string
  /** label text style */
  labelText: string
}

/**
 * This is Dashboard component that displays
 * the everything post in the dashboard.
 *
 * @param {object} props
 * @param props.classes
 * @return component
 */
function Dashboard(props: {
  /** style of dashboard */
  classes: DashboardClassType
}): JSX.Element {
  const { classes } = props
  const dashboardHeaderContent = <DashboardHeader />
  const dashboardPageContent = (
    <React.Fragment>
      <main className={classes.workerRoot}>
        <div className={classes.appBarSpacer} />
        <Typography variant="h6" component="h2" className={classes.labelText}>
          Projects
        </Typography>
        <Divider />
        <DataTable
          dataList={getProjects()}
          headers={[{ header: "Project", align: "left" }]}
        />
      </main>
    </React.Fragment>
  )
  /**
   * render function
   *
   * @return component
   */
  return (
    <HeaderPage
      // TODO: fix this error
      // eslint-disable-next-line react/no-children-prop
      headerContent={dashboardHeaderContent}
      pageContent={dashboardPageContent}
    />
  )
}

/** export Dashboard */
export default withStyles(dashboardStyles)(Dashboard)

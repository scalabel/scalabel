import { Grid, Link } from "@material-ui/core"
import ListItem from "@material-ui/core/ListItem"
import CloseIcon from "@material-ui/icons/Close"
import { withStyles } from "@material-ui/core/styles"
import React from "react"

import { getAuth, getProjects, toProject } from "../common/service"
import { projectListStyle } from "../styles/create"
import { Endpoint } from "../const/connection"

interface ClassType {
  /** style for a colored entry */
  coloredListItem: string
}

// Props for project list
interface ProjectListProps {
  /** Project List classes */
  classes: ClassType
  /** refresh boolean */
  refresh: boolean
}

interface ProjectListState {
  /** projects */
  projectsToExpress: string[]
}

/** Project list sidebar component. Re-renders after
 * submission
 *
 * @param props
 */
class ProjectList extends React.Component<ProjectListProps, ProjectListState> {
  /**
   * Constructor
   *
   * @param props
   */
  public constructor(props: ProjectListProps) {
    super(props)
    this.state = {
      projectsToExpress: getProjects()
    }
  }

  /**
   * renders project list
   */
  public render(): React.ReactNode {
    const { classes } = this.props
    return (
      <div>
        {this.state.projectsToExpress.map((project, index) => (
          <ListItem
            button
            key={project}
            alignItems="center"
            className={index % 2 === 0 ? classes.coloredListItem : ""}
          >
            <Grid container justifyContent="center">
              <Link
                component="button"
                variant="body2"
                color="inherit"
                onClick={() => {
                  toProject(project)
                }}
              >
                {project}
              </Link>
            </Grid>
            {this.state.projectsToExpress[0] !== "No existing project" ? (
              <CloseIcon
                titleAccess="Delete project"
                fontSize="small"
                onClick={() => {
                  if (confirm(`Confirm to delete project: ${project}?`))
                    this.deleteProject(project)
                }}
              />
            ) : null}
          </ListItem>
        ))}
      </div>
    )
  }

  /**
   * This function will delete project according to the project name
   *
   * @param projectName
   */
  public deleteProject(projectName: string): void {
    const xhr = new XMLHttpRequest()
    xhr.onreadystatechange = () => {
      if (xhr.readyState === 4) {
        this.setState({ projectsToExpress: getProjects() })
      }
    }
    xhr.open("GET", `${Endpoint.DELETE_PROJECT}?project_name=` + projectName)
    const auth = getAuth()
    if (auth !== "") {
      xhr.setRequestHeader("Authorization", auth)
    }
    xhr.send()
  }
}

export default withStyles(projectListStyle)(ProjectList)

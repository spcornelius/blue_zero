import helpers
    
if __name__ == "__main__":
    steps_df = helpers.get_steps_df()
    helpers.do_on_task_plot(steps_df)
    helpers.do_off_task_plot(steps_df)

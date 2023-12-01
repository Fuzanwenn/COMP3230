/**
 * Fu Zanwen 3036174194
 * 
 * VScode
 * 
 * Completed everything except for printing process's running status.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/resource.h> 
#include <sys/wait.h>
#include <errno.h>

pid_t last_pid; //The process id of the last running child.

/**
 * Initialize the prompt for JCshell.
*/
void type_prompt() {
    static int first_time = 1;
    if (first_time) {
        const char* CLEAR_SCREEN_ANSI = " \e[1;1H\e[2J";
        write(STDOUT_FILENO, CLEAR_SCREEN_ANSI, 12); //Clear the screen
        first_time = 0;
    }
    printf("## JCshell [%d] ## ", getpid());
}

/**
 * Check the status of the child process.
 * Print the status of the child process if WIFSIGNALED.
*/
void checkStatus(int status) {
    int status_code;
    if (WIFEXITED(status)) {
        status_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        status_code = WTERMSIG(status);
        if (status_code == SIGINT) {
            printf("Interrupt\n");
        } else if (status == SIGSEGV) {
            printf("Segmentation fault\n");
        } else if (status == SIGKILL) {
            printf("Killed\n");
        } else if (status == SIGPIPE) {
            printf("Broken Pipe\n");
        } else {
            printf("No Such Signal\n");
        }
    }
    
}

/**
 * Do nothing but resume the child process.
*/
void siguser(int signal) {
}

void sigint(int signal) {
    printf("## JCshell [%d] ## ", getpid());
    fflush(stdout);
}

/**
 * To handle child process.
*/
void sigchild(int signal) {
    siginfo_t siginfo;

        memset(&siginfo, 0, sizeof(siginfo));
        int ret = waitid(P_ALL, 0, &siginfo, WEXITED | WSTOPPED | WNOHANG | WNOWAIT);
        if (ret == -1) {
            if (errno == ECHILD) {
                return;
            } else {
                perror("waitid");
                exit(EXIT_FAILURE);
            }
        }

        if (siginfo.si_pid == 0) {
            return;
        }

        checkStatus(siginfo.si_status); //Print the status of the child process.

        waitpid(siginfo.si_pid, NULL, 0);

        if (siginfo.si_pid == last_pid) {
            printf("## JCshell [%d] ## ", getpid());
            fflush(stdout);
        } //If the child process is the last running child, print the prompt.
}

/**
 * Execute the command.
*/
void execute(char* command) {
    int command_num = 0; 
    char *commands[5][30]; //Command array to store sequence of commands.
    char *address1, *address2;
    char *command_pointer = strtok_r(command, "|", &address1);
    while (command_num < 5 && command_pointer != NULL) {
        char *arg_pointer;
        int arg_count = 0;

        arg_pointer = strtok_r(command_pointer, " \n\t", &address2);
        while (arg_count < 29 && arg_pointer != NULL) {
            commands[command_num][arg_count] = arg_pointer;
            arg_pointer = strtok_r(NULL, " \n\t", &address2);
            arg_count++;
        }

        commands[command_num][arg_count] = NULL; 
        command_num++;
        command_pointer = strtok_r(NULL, "|", &address1); 
    }

    pid_t command_pids[command_num]; 
    int pipes[4][2];

    for (int i = 0; i < command_num - 1; i++) {
        if (pipe(pipes[i]) < 0) {
            perror("pipe");
            exit(1);
        }
    }

    //Create pipe for each command.
    for (int i = 0; i < command_num; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            if (i != 0) {
                dup2(pipes[i - 1][0], STDIN_FILENO);
            }
            if (i != command_num - 1) {
                dup2(pipes[i][1], STDOUT_FILENO);
            }
            for (int j = 0; j < command_num - 1; j++) {
                close(pipes[j][0]);
                close(pipes[j][1]);
            }
            
            if (execvp(commands[i][0], commands[i]) == -1) {
                perror("");
                exit(1);
            }
            
        } else if (pid < 0) {
            perror("fork");
            exit(1);
        } 
        command_pids[i] = pid;
    }
    last_pid = command_pids[command_num - 1]; //Update the pid of the last child process.

    for (int i = 0; i < command_num; i++) {
        kill(command_pids[i], SIGUSR1);
    }

    for (int j = 0; j < command_num - 1; j++) {
        close(pipes[j][0]);
        close(pipes[j][1]);
    }

}

/**
 * Check if the input is valid.
*/
int isValidInput(char* input) {
    while (*input && (*input == '\t' || *input == ' ')) {
        input++;
    }
    char* last_digit = input + strlen(input) - 1;
    while ((*last_digit == '\t' || *last_digit == ' ' || *last_digit == '\n') && last_digit > input) {
        *last_digit = '\0';
        last_digit--;
    }
    if (strstr(input, "||") || strstr(input, "| |")) {
        return 0; 
    }
    if (*input == '|' || *(input + strlen(input) - 1) == '|') {
        return 0;
    }
    return 1; 
}

int main() {
    char command[1024];

    struct sigaction user_action;
    struct sigaction int_action;
    struct sigaction signal_action;

    user_action.sa_handler = siguser;
    user_action.sa_flags = 0; 
    if (sigaction(SIGUSR1, &user_action, NULL) == -1) {
        perror("sigaction SIGUSR1");
        exit(1);
    }

    int_action.sa_handler = sigint;
    int_action.sa_flags = SA_RESTART | SA_NOCLDSTOP;
    if (sigaction(SIGINT, &int_action, NULL) == -1) {
        perror("sigaction SIGINT");
        exit(1);
    }
    
    signal_action.sa_handler = &sigchild;
    signal_action.sa_flags = SA_RESTART | SA_NOCLDSTOP;
    if (sigaction(SIGCHLD, &signal_action, 0) == -1) {
        perror("sigaction");
        exit(1);
    }
    
    
    type_prompt();

    while (1) {
        fgets(command, 1024, stdin);

        int ln = strlen(command) - 1;
        if (command[ln] == '\n') {
            command[ln] = '\0';
        }
        if (strcmp(command, "exit") == 0) { 
            printf("JCshell: Terminated\n");
            exit(0);
        } else if (strstr(command, "exit") == command && strlen(command) > 4) { 
            printf("JCshell: \"exit\" with other arguments!!!\n");
            continue; 
        } else if (strstr(command, "exit")) {
            printf("JCshell: 'exit': No such file or directory\n");
            continue;
        } else if (!isValidInput(command)) {
            printf("JCshell: should not have two | symbols without in-between command\n");
            continue;
        }
        execute(command);
    }
    return 0;
}
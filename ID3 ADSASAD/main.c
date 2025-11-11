#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "id3.h"

#define MAX_SAMPLES 150
#define MAX_COLS 5
#define MAX_LINE 256
#define MAX_DATASET (MAX_SAMPLES * MAX_COLS)

// discretize continuous values into categories
char* discretize_value(float value, float low, float high) {
    float third = low + (high - low) / 3.0;
    float two_thirds = low + 2.0 * (high - low) / 3.0;
    
    if (value < third) return "LOW";
    if (value < two_thirds) return "MED";
    return "HIGH";
}

int main()
{
    FILE *file = NULL;
    char line[MAX_LINE];
    char *data_set[MAX_DATASET + 1];
    int row_count = 0;
    int col_count = 5;
    int result = 0;
    int i, j;
    
    // Min and max values for each attribute (sepal_length, sepal_width, petal_length, petal_width)
    float attr_min[4] = {4.3, 2.0, 1.0, 0.1};
    float attr_max[4] = {7.9, 4.4, 6.9, 2.5};
    
    // String array for column headers
    char *column_names[MAX_COLS] = { '\0' };
    
    column_names[0] = "SEPAL_LENGTH";
    column_names[1] = "SEPAL_WIDTH";
    column_names[2] = "PETAL_LENGTH";
    column_names[3] = "PETAL_WIDTH";
    column_names[4] = "SPECIES";
    
    printf("Reading iris.csv...\n");
    
    // Open the CSV file
    file = fopen("../iris.csv", "r");
    if (file == NULL) {
        printf("Error: Could not open iris.csv\n");
        return -1;
    }
    
    // Skip header line
    if (fgets(line, MAX_LINE, file) == NULL) {
        printf("Error: Empty file\n");
        fclose(file);
        return -1;
    }
    
    // Read data from CSV
    while (fgets(line, MAX_LINE, file) != NULL && row_count < MAX_SAMPLES) {
        char *token;
        char line_copy[MAX_LINE];
        strcpy(line_copy, line);
        
        // Remove newline
        if (line_copy[strlen(line_copy) - 1] == '\n') {
            line_copy[strlen(line_copy) - 1] = '\0';
        }
        
        // Parse CSV line
        float values[4];
        char species[32];
        
        if (sscanf(line_copy, "%f,%f,%f,%f,%31s", 
                   &values[0], &values[1], &values[2], &values[3], species) == 5) {
            
            // Remove any trailing whitespace/newline from species
            int species_len = strlen(species);
            while (species_len > 0 && (species[species_len - 1] == '\n' || 
                   species[species_len - 1] == '\r' || species[species_len - 1] == ' ')) {
                species[species_len - 1] = '\0';
                species_len--;
            }
            
            // Discretize numeric values and prefix with column name
            for (j = 0; j < 4; j++) {
                char *category = discretize_value(values[j], attr_min[j], attr_max[j]);
                char prefixed[64];
                // Create unique value by prefixing with column name
                sprintf(prefixed, "%s_%s", column_names[j], category);
                data_set[row_count * col_count + j] = malloc(strlen(prefixed) + 1);
                strcpy(data_set[row_count * col_count + j], prefixed);
            }
            
            // Add species (already categorical)
            data_set[row_count * col_count + 4] = malloc(strlen(species) + 1);
            strcpy(data_set[row_count * col_count + 4], species);
            
            row_count++;
        }
    }
    
    fclose(file);
    
    // Null terminate dataset
    data_set[row_count * col_count] = NULL;
    
    printf("Loaded %d samples from iris.csv\n\n", row_count);
    
    // Call ID3 algorithm
    result = id3_get_rules(
        data_set,               // pointer to data set
        col_count,              // total columns : attributes + last column (classification)
        row_count,              // total database samples (rows)
        column_names );
    
    printf("\nSearch result (%d) : ", result);
    if (result == 0) {
        printf("OK\n");
    } else {
        printf("Error memory allocation\n");
    }
    
    // Free allocated memory
    for (i = 0; i < row_count * col_count; i++) {
        if (data_set[i] != NULL) {
            free(data_set[i]);
        }
    }
    
    return result;
}

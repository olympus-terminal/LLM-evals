#!/bin/bash

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counting algal hits
echo -e "${GREEN}Counting algal hits...${NC}"
fgrep -c '@' eval-resu*

echo -e "\n${GREEN}Counting bacterial hits...${NC}"
fgrep -c '!' eval-resu*

echo -e "\n${YELLOW}In other words, from the algal holdout set there are:${NC}"
count_algae=$(fgrep -c '@' eval-results_Filtered_algal_doubled.aa.fa.unwrapped_no_headers.wrapped.10)
echo -e "${GREEN}${count_algae} algal signatures.${NC}"

count_bacteria=$(fgrep -c '!' eval-results_Filtered_algal_doubled.aa.fa.unwrapped_no_headers.wrapped.10)
echo -e "${RED}${count_bacteria} bacterial signatures.${NC}"

echo -e "\n${YELLOW}And from the bacterial holdout set, there are:${NC}"
count_algae_bact=$(fgrep -c '@' eval-results_bact1_accns.headers-fetched.aa.fa.unwrapped_no_headers.wrapped.10)
echo -e "${GREEN}${count_algae_bact} algal signatures.${NC}"

count_bacteria_bact=$(fgrep -c '!' eval-results_bact1_accns.headers-fetched.aa.fa.unwrapped_no_headers.wrapped.10)
echo -e "${RED}${count_bacteria_bact} bacterial signatures.${NC}"

# Calculate performance metrics
total_algae=$((count_algae + count_algae_bact))
total_bacteria=$((count_bacteria + count_bacteria_bact))

true_positives_algae=$count_algae
false_positives_algae=$count_bacteria
true_negatives_bacteria=$count_bacteria_bact
false_negatives_bacteria=$count_algae_bact

precision_algae=$(echo "scale=4; $true_positives_algae / ($true_positives_algae + $false_positives_algae)" | bc)
recall_algae=$(echo "scale=4; $true_positives_algae / ($true_positives_algae + $false_negatives_bacteria)" | bc)
f1_score_algae=$(echo "scale=4; 2 * ($precision_algae * $recall_algae) / ($precision_algae + $recall_algae)" | bc)

precision_bacteria=$(echo "scale=4; $true_negatives_bacteria / ($true_negatives_bacteria + $false_negatives_bacteria)" | bc)
recall_bacteria=$(echo "scale=4; $true_negatives_bacteria / ($true_negatives_bacteria + $false_positives_algae)" | bc)
f1_score_bacteria=$(echo "scale=4; 2 * ($precision_bacteria * $recall_bacteria) / ($precision_bacteria + $recall_bacteria)" | bc)

echo -e "\n${YELLOW}Performance Metrics:${NC}"
echo -e "Algal Precision: ${GREEN}$precision_algae${NC}"
echo -e "Algal Recall: ${GREEN}$recall_algae${NC}"
echo -e "Algal F1 Score: ${GREEN}$f1_score_algae${NC}"

echo -e "Bacterial Precision: ${RED}$precision_bacteria${NC}"
echo -e "Bacterial Recall: ${RED}$recall_bacteria${NC}"
echo -e "Bacterial F1 Score: ${RED}$f1_score_bacteria${NC}"

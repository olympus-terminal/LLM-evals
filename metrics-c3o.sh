## after next-token generation for prediction, before data collation to csv

#!/bin/bash

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if the correct number of arguments is provided
if [ $# -ne 2 ]; then
  echo "Usage: $0 <algal_holdout_file> <bacterial_holdout_file>"
  exit 1
fi

algal_holdout_file=$1
bacterial_holdout_file=$2

# Counting algal hits
echo -e "${GREEN}Counting algal hits...${NC}"
count_algae=$(fgrep -c '@' "$algal_holdout_file")
echo -e "${GREEN}${count_algae} algal signatures.${NC}"

# Counting bacterial hits
echo -e "\n${GREEN}Counting bacterial hits...${NC}"
count_bacteria=$(fgrep -c '!' "$algal_holdout_file")
echo -e "${RED}${count_bacteria} bacterial signatures.${NC}"

echo -e "\n${YELLOW}In the algal holdout set, there are:${NC}"
echo -e "${GREEN}${count_algae} algal signatures.${NC}"
echo -e "${RED}${count_bacteria} bacterial signatures.${NC}"

echo -e "\n${YELLOW}And in the bacterial holdout set, there are:${NC}"
count_algae_bact=$(fgrep -c '@' "$bacterial_holdout_file")
echo -e "${GREEN}${count_algae_bact} algal signatures.${NC}"

count_bacteria_bact=$(fgrep -c '!' "$bacterial_holdout_file")
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

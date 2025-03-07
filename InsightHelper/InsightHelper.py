def format_insights_for_llm(insights):
    """
    Compresses data insights while minimizing information loss.
    Uses efficient formatting and prioritizes critical information.
    
    Compression strategies:
    1. Rounds numbers to 2 decimal places
    2. Removes redundant information
    3. Uses compact formatting
    4. Prioritizes statistically significant findings
    5. Summarizes repeated patterns
    """
    def round_numbers(value):
        """Round numeric values to 2 decimal places"""
        try:
            return round(float(value), 2)
        except (ValueError, TypeError):
            return value

    def format_dict_compact(d):
        """Convert dictionary to compact string representation"""
        return '|'.join(f"{k}:{v}" for k, v in d.items())

    context_parts = []
    
    # Descriptive Statistics - Keep only essential metrics
    if "descriptive" in insights:
        desc = insights["descriptive"]
        stats = []
        
        # Combine distribution metrics
        dist_metrics = {}
        for key in desc:
            if key.endswith(('_kurtosis', '_skewness')):
                var_name = key.split('_')[0]
                metric = key.split('_')[1]
                if var_name not in dist_metrics:
                    dist_metrics[var_name] = {}
                dist_metrics[var_name][metric] = round_numbers(desc[key])
        
        if dist_metrics:
            stats.append("DIST:" + ','.join(
                f"{var}({format_dict_compact(metrics)})"
                for var, metrics in dist_metrics.items()
            ))
        
        # Compress basic stats
        if "basic_stats" in desc:
            basic = []
            for var, var_stats in desc["basic_stats"].items():
                rounded_stats = {k: round_numbers(v) for k, v in var_stats.items()}
                basic.append(f"{var}({format_dict_compact(rounded_stats)})")
            if basic:
                stats.append("BASIC:" + ','.join(basic))
        
        # Data quality metrics
        quality_metrics = {}
        if "data_types" in desc:
            quality_metrics["types"] = desc["data_types"]
        if "missing_values" in desc:
            quality_metrics["missing"] = {k: v for k, v in desc["missing_values"].items() if v > 0}
        if "unique_values" in desc:
            quality_metrics["unique"] = desc["unique_values"]
        
        if quality_metrics:
            stats.append("QUAL:" + format_dict_compact(quality_metrics))
        
        if stats:
            context_parts.extend(stats)
    
    # Diagnostic Analytics - Focus on strong correlations
    if "diagnostic" in insights:
        diag = insights["diagnostic"]
        if "strong_correlations" in diag:
            strong = []
            for corr in diag["strong_correlations"]:
                corr_val = round_numbers(corr.get('correlation', 0))
                if abs(corr_val) >= 0.5:  # Only include significant correlations
                    strong.append(f"{'+'.join(corr['variables'])}={corr_val}")
            if strong:
                context_parts.append("CORR:" + ','.join(strong))
    
    # Outliers - Summarize counts and significant examples
    if "outliers" in insights:
        out = insights["outliers"]
        outlier_info = []
        
        if "summary" in out:
            summary = out["summary"]
            outlier_info.append(
                f"SUMMARY:rows={summary.get('total_rows', 0)}|"
                f"iqr%={round_numbers(summary.get('iqr_outlier_percentage', 0))}|"
                f"z%={round_numbers(summary.get('z_score_outlier_percentage', 0))}"
            )
        
        # Include only top outliers
        for method in ['iqr_outliers', 'z_score_outliers']:
            if method in out:
                data = out[method]
                count = data.get('count', 0)
                if count > 0:
                    values = data.get('values', [])[:3]  # Keep only top 3 outliers
                    outlier_info.append(
                        f"{method.upper()}:count={count}|"
                        f"examples={','.join(format_dict_compact(v) for v in values)}"
                    )
        
        if outlier_info:
            context_parts.extend(outlier_info)
    
    # Prescriptive Analytics - Keep only high-impact opportunities
    if "prescriptive" in insights:
        pres = insights["prescriptive"]
        if "optimization_opportunities" in pres:
            opps = []
            for opp in pres["optimization_opportunities"]:
                if abs(float(opp.get('correlation', 0))) >= 0.7:  # Only strong correlations
                    opps.append(
                        f"{'+'.join(opp['variables'])}:"
                        f"{round_numbers(opp['correlation'])}:"
                        f"{opp['suggestion'][:100]}"  # Truncate long suggestions
                    )
            if opps:
                context_parts.append("OPT:" + '|'.join(opps))
    
    return "\n".join(context_parts)

def generate_insight_llm_prompt(insights_context):
    """
    Generate a structured and professional prompt for the LLM based on provided data insights.
    """
    system_instructions = (
        "You are a data scientist and data interpretation expert tasked with interpreting ,explaining data insights and creating a diagnostic report, "
        "and preparing a professional, comprehensive diagnostic report. Follow these guidelines:\n"
        "1. Provide an analysis of the data insights.\n"
        "2. Identify and explain key patterns, trends, and relationships in the data.\n"
        "3. Translate statistical findings into plain language.\n"
        "4. Include specific numbers and metrics where relevant.\n"
        "5. Highlight potential data quality issues or limitations.\n"
        "6. Suggest actionable recommendations and business implications.\n/"
    )

    user_request = (
        f"Please analyze the following data insights:\n\n{insights_context}\n\n"
        "Read the data carefully and generate a comprehensive report that includes. Word limit is 800 words.\n"
        "First provide 2-3 line summary of the CSV file. Include the names of every headers of csv."
        "1. A short description of key findings with exact numbers. Use every single data point and explain what they mean in simple words to a layman. Use bullet points. Also incldue a point include a line explaining the gist of key findings to a layman.\n"
        "2. Significant patterns, relationships, or trends with brief explanation.\n"
        "3. Highlights of outliers or unusual patterns in short, or a mention if none exist. Use every data point to explain why the value is outliers or unusual in short using simple language.\n"
        "4. Generate Actionable recommendations based on the insights and keep it simple and applicable in real world.\n"
        "5. Predictions or forecasts derived from the data if applicable.\n"
        "5. Diagnostic analysis of the data based on the insights.\n"
        "6. Recommendations for the best visualization methods to represent trends and "
        "insights, specifying the features and variable names to use. Maximum 2-3"
    )

    return {
        "messages": [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_request},
        ]
    }
